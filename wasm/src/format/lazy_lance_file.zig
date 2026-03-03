//! Lazy Lance file reader - Column-first I/O
//!
//! Only reads the bytes needed for requested columns, not the entire file.
//! This provides 10-100x I/O improvement for queries that only need 1-2 columns
//! from a wide table (e.g., reading 1MB instead of 149MB).
//!
//! ## I/O Pattern
//! 1. Read footer (40 bytes) - get file layout
//! 2. Read column metadata offset table (~KB) - locate columns
//! 3. Read specific column's metadata - get data buffer locations
//! 4. Read only that column's data buffers
//!
//! ## Usage
//! ```zig
//! var file_reader = try FileReader.open("data.lance");
//! defer file_reader.close();
//!
//! var lazy = try LazyLanceFile.init(allocator, file_reader.reader());
//! defer lazy.deinit();
//!
//! // Only reads the bytes for column 1
//! const amounts = try lazy.readFloat64Column(1);
//! defer allocator.free(amounts);
//! ```

const std = @import("std");
const footer_mod = @import("footer.zig");
const page_row_index_mod = @import("page_row_index.zig");
const proto = @import("querymode.proto");
const encoding = @import("querymode.encoding");
const io = @import("querymode.io");

const Footer = footer_mod.Footer;
const FOOTER_SIZE = footer_mod.FOOTER_SIZE;
const Reader = io.Reader;
const ReadError = io.ReadError;
const ByteRange = io.ByteRange;
const BatchReadResult = io.BatchReadResult;
const ColumnMetadata = proto.ColumnMetadata;
const PlainDecoder = encoding.PlainDecoder;
const PageRowIndex = page_row_index_mod.PageRowIndex;

pub const LazyLanceFileError = error{
    FileTooSmall,
    InvalidMagic,
    UnsupportedVersion,
    InvalidMetadata,
    OutOfMemory,
    ColumnOutOfBounds,
    NoPages,
    IoError,
};

/// Position and length pair from offset tables
pub const OffsetEntry = struct {
    position: u64,
    length: u64,
};

/// Lazy Lance file reader - only reads requested columns
pub const LazyLanceFile = struct {
    allocator: std.mem.Allocator,
    reader: Reader,
    footer: Footer,
    file_size: u64,

    /// Column metadata offset table (cached after first access)
    column_meta_entries: ?[]OffsetEntry = null,

    const Self = @This();

    /// Initialize lazy reader - only reads footer (40 bytes)
    pub fn init(allocator: std.mem.Allocator, reader: Reader) LazyLanceFileError!Self {
        const file_size = reader.size() catch return LazyLanceFileError.IoError;

        if (file_size < FOOTER_SIZE) {
            return LazyLanceFileError.FileTooSmall;
        }

        // Read footer (40 bytes from end)
        var footer_buf: [FOOTER_SIZE]u8 = undefined;
        reader.readExact(file_size - FOOTER_SIZE, &footer_buf) catch {
            return LazyLanceFileError.IoError;
        };

        const footer = Footer.parse(&footer_buf) catch {
            return LazyLanceFileError.InvalidMagic;
        };

        if (!footer.isSupported()) {
            return LazyLanceFileError.UnsupportedVersion;
        }

        return Self{
            .allocator = allocator,
            .reader = reader,
            .footer = footer,
            .file_size = file_size,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.column_meta_entries) |entries| {
            self.allocator.free(entries);
        }
    }

    /// Get number of columns
    pub fn numColumns(self: Self) u32 {
        return self.footer.num_columns;
    }

    /// Load column metadata offset table (lazy - only on first use)
    fn ensureColumnMetaEntries(self: *Self) LazyLanceFileError![]OffsetEntry {
        if (self.column_meta_entries) |entries| {
            return entries;
        }

        const num_cols = self.footer.num_columns;
        const table_size = num_cols * 16; // 16 bytes per entry
        const table_offset = self.footer.column_meta_offsets_start;

        // Read column metadata offset table
        const table_buf = self.allocator.alloc(u8, table_size) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        defer self.allocator.free(table_buf);

        self.reader.readExact(table_offset, table_buf) catch {
            return LazyLanceFileError.IoError;
        };

        // Parse offset entries
        var entries = self.allocator.alloc(OffsetEntry, num_cols) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        errdefer self.allocator.free(entries);

        var i: u32 = 0;
        while (i < num_cols) : (i += 1) {
            const entry_pos = i * 16;
            entries[i] = .{
                .position = std.mem.readInt(u64, table_buf[entry_pos..][0..8], .little),
                .length = std.mem.readInt(u64, table_buf[entry_pos + 8 ..][0..8], .little),
            };
        }

        self.column_meta_entries = entries;
        return entries;
    }

    /// Read column metadata for a specific column
    fn readColumnMetadata(self: *Self, col_idx: u32) LazyLanceFileError!ColumnMetadata {
        const entries = try self.ensureColumnMetaEntries();

        if (col_idx >= entries.len) {
            return LazyLanceFileError.ColumnOutOfBounds;
        }

        const entry = entries[col_idx];

        // Read column metadata bytes
        const meta_buf = self.allocator.alloc(u8, entry.length) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        defer self.allocator.free(meta_buf);

        self.reader.readExact(entry.position, meta_buf) catch {
            return LazyLanceFileError.IoError;
        };

        // Parse column metadata
        return ColumnMetadata.parse(self.allocator, meta_buf) catch {
            return LazyLanceFileError.InvalidMetadata;
        };
    }

    /// Generic helper to read numeric column data
    /// Handles both nullable (2+ buffers) and non-nullable (1 buffer) columns
    fn readNumericColumn(self: *Self, comptime T: type, col_idx: u32) LazyLanceFileError![]T {
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return LazyLanceFileError.NoPages;

        var total_values: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len > 0) {
                // For nullable columns (2+ buffers): data is in buffer[1]
                // For non-nullable columns (1 buffer): data is in buffer[0]
                const data_buf_idx: usize = if (page.buffer_sizes.len >= 2) 1 else 0;
                total_values += page.buffer_sizes[data_buf_idx] / @sizeOf(T);
            }
        }

        var result = self.allocator.alloc(T, total_values) catch return LazyLanceFileError.OutOfMemory;
        errdefer self.allocator.free(result);

        var offset: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len == 0 or page.buffer_sizes.len == 0) continue;

            // For nullable columns (2+ buffers): data is in buffer[1]
            // For non-nullable columns (1 buffer): data is in buffer[0]
            const data_buf_idx: usize = if (page.buffer_sizes.len >= 2) 1 else 0;

            const buffer_data = self.allocator.alloc(u8, page.buffer_sizes[data_buf_idx]) catch return LazyLanceFileError.OutOfMemory;
            defer self.allocator.free(buffer_data);

            self.reader.readExact(page.buffer_offsets[data_buf_idx], buffer_data) catch return LazyLanceFileError.IoError;

            const decoder = PlainDecoder.init(buffer_data);
            const page_values = (switch (T) {
                f64 => decoder.readAllFloat64(self.allocator),
                i64 => decoder.readAllInt64(self.allocator),
                f32 => decoder.readAllFloat32(self.allocator),
                i32 => decoder.readAllInt32(self.allocator),
                else => @compileError("unsupported type"),
            }) catch return LazyLanceFileError.OutOfMemory;
            defer self.allocator.free(page_values);

            @memcpy(result[offset .. offset + page_values.len], page_values);
            offset += page_values.len;
        }

        return result;
    }

    pub fn readFloat64Column(self: *Self, col_idx: u32) LazyLanceFileError![]f64 {
        return self.readNumericColumn(f64, col_idx);
    }

    pub fn readInt64Column(self: *Self, col_idx: u32) LazyLanceFileError![]i64 {
        return self.readNumericColumn(i64, col_idx);
    }

    /// Read validity bitmap for a nullable column
    /// Returns null if column has no validity bitmap (all values valid)
    /// The bitmap uses Arrow format: bit 1 = valid, bit 0 = null
    pub fn readValidityBitmap(self: *Self, col_idx: u32) LazyLanceFileError!?[]u8 {
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return LazyLanceFileError.NoPages;

        // Check if column has validity bitmap by looking at buffer count
        // Nullable columns have 2+ buffers: [validity_bitmap, data, ...]
        // Non-nullable columns have 1 buffer: [data]
        var has_validity = false;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len >= 2) {
                has_validity = true;
                break;
            }
        }

        if (!has_validity) return null;

        // Calculate total bitmap size
        var total_bytes: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len >= 2) {
                total_bytes += page.buffer_sizes[0];
            }
        }

        if (total_bytes == 0) return null;

        var result = self.allocator.alloc(u8, total_bytes) catch return LazyLanceFileError.OutOfMemory;
        errdefer self.allocator.free(result);

        var offset: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len >= 2 and page.buffer_sizes.len >= 2) {
                const bitmap_size = page.buffer_sizes[0];
                if (bitmap_size > 0) {
                    self.reader.readExact(page.buffer_offsets[0], result[offset .. offset + bitmap_size]) catch return LazyLanceFileError.IoError;
                    offset += bitmap_size;
                }
            }
        }

        return result;
    }

    /// Get bytes read stats (for benchmarking)
    pub fn getBytesRead(self: Self, col_idx: u32) LazyLanceFileError!u64 {
        var total: u64 = FOOTER_SIZE; // Footer always read

        const entries = self.column_meta_entries orelse {
            // Would need to read offset table
            total += self.footer.num_columns * 16;
            return total;
        };

        if (col_idx >= entries.len) {
            return LazyLanceFileError.ColumnOutOfBounds;
        }

        // Add offset table size
        total += self.footer.num_columns * 16;

        // Add column metadata size
        const entry = entries[col_idx];
        total += entry.length;

        // Would need to parse column metadata to get exact data size
        // For now, return metadata + offset table
        return total;
    }

    /// Get schema bytes from global buffer 0 (lazy read)
    pub fn getSchemaBytes(self: *Self) LazyLanceFileError!?[]const u8 {
        if (self.footer.num_global_buffers == 0) return null;

        // Read global buffer offset table (16 bytes per entry)
        const gbo_start = self.footer.global_buff_offsets_start;
        var entry_buf: [16]u8 = undefined;
        self.reader.readExact(gbo_start, &entry_buf) catch {
            return LazyLanceFileError.IoError;
        };

        const position = std.mem.readInt(u64, entry_buf[0..8], .little);
        const length = std.mem.readInt(u64, entry_buf[8..16], .little);

        // Read schema bytes
        const schema_bytes = self.allocator.alloc(u8, length) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        self.reader.readExact(position, schema_bytes) catch {
            self.allocator.free(schema_bytes);
            return LazyLanceFileError.IoError;
        };

        return schema_bytes;
    }

    pub fn readInt32Column(self: *Self, col_idx: u32) LazyLanceFileError![]i32 {
        return self.readNumericColumn(i32, col_idx);
    }

    pub fn readFloat32Column(self: *Self, col_idx: u32) LazyLanceFileError![]f32 {
        return self.readNumericColumn(f32, col_idx);
    }

    /// Read all bool values from a column (column-first I/O)
    pub fn readBoolColumn(self: *Self, col_idx: u32) LazyLanceFileError![]bool {
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return LazyLanceFileError.NoPages;

        var total_values: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len > 0) {
                // Bool stored as 1 byte each
                total_values += page.buffer_sizes[0];
            }
        }

        var result = self.allocator.alloc(bool, total_values) catch {
            return LazyLanceFileError.OutOfMemory;
        };
        errdefer self.allocator.free(result);

        var offset: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len == 0 or page.buffer_sizes.len == 0) continue;

            const buffer_offset = page.buffer_offsets[0];
            const buffer_size = page.buffer_sizes[0];

            const buffer_data = self.allocator.alloc(u8, buffer_size) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(buffer_data);

            self.reader.readExact(buffer_offset, buffer_data) catch {
                return LazyLanceFileError.IoError;
            };

            const decoder = PlainDecoder.init(buffer_data);
            const page_values = decoder.readAllBool(self.allocator) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(page_values);

            @memcpy(result[offset .. offset + page_values.len], page_values);
            offset += page_values.len;
        }

        return result;
    }

    /// Read all string values from a column (column-first I/O)
    pub fn readStringColumn(self: *Self, col_idx: u32) LazyLanceFileError![][]const u8 {
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return LazyLanceFileError.NoPages;

        // Strings use buffers 0 (offsets) and 1 (data)
        var all_strings = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (all_strings.items) |s| self.allocator.free(s);
            all_strings.deinit();
        }

        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len < 2 or page.buffer_sizes.len < 2) continue;

            const offsets_pos = page.buffer_offsets[0];
            const offsets_size = page.buffer_sizes[0];
            const data_pos = page.buffer_offsets[1];
            const data_size = page.buffer_sizes[1];

            // Read offsets buffer
            const offsets_buf = self.allocator.alloc(u8, offsets_size) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(offsets_buf);

            self.reader.readExact(offsets_pos, offsets_buf) catch {
                return LazyLanceFileError.IoError;
            };

            // Read data buffer
            const data_buf = self.allocator.alloc(u8, data_size) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(data_buf);

            self.reader.readExact(data_pos, data_buf) catch {
                return LazyLanceFileError.IoError;
            };

            // Decode strings
            const decoder = PlainDecoder.init(offsets_buf);
            const strings = decoder.readAllStrings(self.allocator, data_buf) catch {
                return LazyLanceFileError.OutOfMemory;
            };
            defer self.allocator.free(strings);

            for (strings) |s| {
                const copy = self.allocator.dupe(u8, s) catch {
                    return LazyLanceFileError.OutOfMemory;
                };
                all_strings.append(copy) catch {
                    self.allocator.free(copy);
                    return LazyLanceFileError.OutOfMemory;
                };
            }
        }

        return all_strings.toOwnedSlice() catch {
            return LazyLanceFileError.OutOfMemory;
        };
    }

    // ========================================================================
    // Selective Row Reading (Late Materialization Support)
    // ========================================================================

    /// Default gap threshold for range coalescing (4KB)
    const DEFAULT_GAP_THRESHOLD: u64 = 4096;

    /// Read numeric column values at specific row indices.
    /// Uses PageRowIndex for efficient byte offset mapping and batch reads.
    ///
    /// This is the key method for late materialization - it only reads the
    /// bytes needed for the requested rows, not the entire column.
    fn readNumericColumnAtIndices(self: *Self, comptime T: type, col_idx: u32, row_indices: []const u32) LazyLanceFileError![]T {
        if (row_indices.len == 0) {
            return self.allocator.alloc(T, 0) catch return LazyLanceFileError.OutOfMemory;
        }

        // Get column metadata
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return LazyLanceFileError.NoPages;

        // Build page row index
        var page_idx = PageRowIndex.init(self.allocator, &col_meta) catch return LazyLanceFileError.OutOfMemory;
        defer page_idx.deinit();

        // Determine data buffer index (nullable columns have validity bitmap in buffer 0)
        const data_buf_idx: usize = if (col_meta.pages.len > 0 and col_meta.pages[0].buffer_sizes.len >= 2) 1 else 0;

        // Get coalesced byte ranges
        const format_ranges = page_idx.getByteRanges(row_indices, @sizeOf(T), DEFAULT_GAP_THRESHOLD, data_buf_idx) catch return LazyLanceFileError.OutOfMemory;
        defer page_idx.allocator.free(format_ranges);

        if (format_ranges.len == 0) {
            return self.allocator.alloc(T, 0) catch return LazyLanceFileError.OutOfMemory;
        }

        // Convert to reader's ByteRange type
        const reader_ranges = self.allocator.alloc(ByteRange, format_ranges.len) catch return LazyLanceFileError.OutOfMemory;
        defer self.allocator.free(reader_ranges);

        for (format_ranges, 0..) |fr, i| {
            reader_ranges[i] = ByteRange{
                .start = fr.start,
                .end = fr.end,
            };
        }

        // Batch read all ranges
        var batch_result = self.reader.batchRead(self.allocator, reader_ranges) catch return LazyLanceFileError.IoError;
        defer batch_result.deinit();

        // Allocate result array
        const result = self.allocator.alloc(T, row_indices.len) catch return LazyLanceFileError.OutOfMemory;
        errdefer self.allocator.free(result);

        // Map each row to its value from the batch results
        const value_size = @sizeOf(T);
        for (row_indices, 0..) |row, result_idx| {
            const loc = page_idx.getFixedRowOffset(row, value_size, data_buf_idx) orelse {
                // Row out of bounds - fill with zero
                result[result_idx] = @as(T, 0);
                continue;
            };

            // Find which range contains this byte offset
            var found = false;
            for (format_ranges, 0..) |range, range_idx| {
                if (loc.byte_offset >= range.start and loc.byte_offset < range.end) {
                    const buf_offset: usize = @intCast(loc.byte_offset - range.start);
                    const buf = batch_result.buffers[range_idx];

                    if (buf_offset + value_size <= buf.len) {
                        // Read value based on type
                        if (T == f64) {
                            result[result_idx] = @bitCast(std.mem.readInt(u64, buf[buf_offset..][0..8], .little));
                        } else if (T == f32) {
                            result[result_idx] = @bitCast(std.mem.readInt(u32, buf[buf_offset..][0..4], .little));
                        } else if (T == i64 or T == u64) {
                            result[result_idx] = @bitCast(std.mem.readInt(u64, buf[buf_offset..][0..8], .little));
                        } else if (T == i32 or T == u32) {
                            result[result_idx] = @bitCast(std.mem.readInt(u32, buf[buf_offset..][0..4], .little));
                        } else {
                            @compileError("unsupported type for selective read");
                        }
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                result[result_idx] = @as(T, 0);
            }
        }

        return result;
    }

    /// Read int64 values at specific row indices.
    /// Only fetches the bytes needed for the requested rows.
    pub fn readInt64ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) LazyLanceFileError![]i64 {
        return self.readNumericColumnAtIndices(i64, col_idx, row_indices);
    }

    /// Read float64 values at specific row indices.
    /// Only fetches the bytes needed for the requested rows.
    pub fn readFloat64ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) LazyLanceFileError![]f64 {
        return self.readNumericColumnAtIndices(f64, col_idx, row_indices);
    }

    /// Read int32 values at specific row indices.
    /// Only fetches the bytes needed for the requested rows.
    pub fn readInt32ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) LazyLanceFileError![]i32 {
        return self.readNumericColumnAtIndices(i32, col_idx, row_indices);
    }

    /// Read float32 values at specific row indices.
    /// Only fetches the bytes needed for the requested rows.
    pub fn readFloat32ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) LazyLanceFileError![]f32 {
        return self.readNumericColumnAtIndices(f32, col_idx, row_indices);
    }

    /// Build a PageRowIndex for a column.
    /// Caller owns the returned index and must call deinit().
    /// Useful for performing multiple selective reads on the same column.
    pub fn buildPageRowIndex(self: *Self, col_idx: u32) LazyLanceFileError!PageRowIndex {
        var col_meta = try self.readColumnMetadata(col_idx);
        defer col_meta.deinit(self.allocator);

        return PageRowIndex.init(self.allocator, &col_meta) catch return LazyLanceFileError.OutOfMemory;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "lazy lance file error enum" {
    const err: LazyLanceFileError = LazyLanceFileError.FileTooSmall;
    try std.testing.expect(err == LazyLanceFileError.FileTooSmall);
}
