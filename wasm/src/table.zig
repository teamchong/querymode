//! High-level Table API for reading Lance files.
//!
//! This module provides a convenient interface for reading Lance files,
//! combining file parsing, schema access, and column value decoding.

const std = @import("std");
const format = @import("querymode.format");
const proto = @import("querymode.proto");
const encoding = @import("querymode.encoding");
const simd = @import("simd");
const io = @import("querymode.io");

const LanceFile = format.LanceFile;
const LazyLanceFile = format.LazyLanceFile;
const FileReader = io.FileReader;
const MmapReader = io.MmapReader;
const Schema = proto.Schema;
const Field = proto.Field;
const ColumnMetadata = proto.ColumnMetadata;
const PlainDecoder = encoding.PlainDecoder;

/// Errors that can occur when reading a table.
pub const TableError = error{
    NoSchema,
    ColumnNotFound,
    InvalidMetadata,
    UnsupportedType,
    OutOfMemory,
    FileTooSmall,
    InvalidMagic,
    UnsupportedVersion,
    ReadError,
    ColumnOutOfBounds,
    NoPages,
    InvalidBufferIndex,
    IndexOutOfBounds,
    DictionaryEncodingNotSupported,
};

/// High-level table reader for Lance files.
pub const Table = struct {
    allocator: std.mem.Allocator,
    lance_file: LanceFile,
    schema: ?Schema,

    const Self = @This();

    /// Open a table from a byte slice.
    pub fn init(allocator: std.mem.Allocator, data: []const u8) TableError!Self {
        var lance_file = LanceFile.init(allocator, data) catch |err| {
            return switch (err) {
                error.FileTooSmall => TableError.FileTooSmall,
                error.InvalidMagic => TableError.InvalidMagic,
                error.UnsupportedVersion => TableError.UnsupportedVersion,
                error.InvalidMetadata => TableError.InvalidMetadata,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
        errdefer lance_file.deinit();

        // Parse schema from global buffer 0
        var schema: ?Schema = null;
        if (lance_file.getSchemaBytes()) |schema_bytes| {
            schema = Schema.parse(allocator, schema_bytes) catch null;
        }

        return Self{
            .allocator = allocator,
            .lance_file = lance_file,
            .schema = schema,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.schema) |*s| s.deinit();
        self.lance_file.deinit();
    }

    /// Get the number of columns.
    pub fn numColumns(self: Self) u32 {
        return self.lance_file.numColumns();
    }

    /// Get column names from schema.
    pub fn columnNames(self: Self) TableError![][]const u8 {
        const schema = self.schema orelse return TableError.NoSchema;
        return schema.columnNames(self.allocator) catch return TableError.OutOfMemory;
    }

    /// Get the schema.
    pub fn getSchema(self: Self) ?Schema {
        return self.schema;
    }

    /// Get column index by name (returns field array index).
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        const schema = self.schema orelse return null;
        return schema.fieldIndex(name);
    }

    /// Get physical column ID by name (for use with column metadata).
    pub fn physicalColumnId(self: Self, name: []const u8) ?u32 {
        const schema = self.schema orelse return null;
        return schema.physicalColumnId(name);
    }

    /// Get field info by column index (array index in schema.fields).
    pub fn getField(self: Self, col_idx: usize) ?Field {
        const schema = self.schema orelse return null;
        if (col_idx >= schema.fields.len) return null;
        return schema.fields[col_idx];
    }

    /// Get field info by physical column ID.
    pub fn getFieldById(self: Self, field_id: u32) ?Field {
        const schema = self.schema orelse return null;
        for (schema.fields) |field| {
            if (field.id >= 0 and @as(u32, @intCast(field.id)) == field_id) {
                return field;
            }
        }
        return null;
    }

    /// Get row count for a column.
    pub fn rowCount(self: Self, col_idx: u32) TableError!u64 {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        return col_meta.rowCount();
    }

    // ========================================================================
    // Typed Column Readers
    // ========================================================================

    /// Generic helper to read numeric column data
    fn readNumericColumn(self: Self, comptime T: type, col_idx: u32) TableError![]T {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch return TableError.ColumnOutOfBounds;
        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch return TableError.InvalidMetadata;
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;

        var total_values: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len > 0) total_values += page.buffer_sizes[0] / @sizeOf(T);
        }

        const result = self.allocator.alloc(T, total_values) catch return TableError.OutOfMemory;
        errdefer self.allocator.free(result);

        var offset: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len == 0 or page.buffer_sizes.len == 0) continue;
            const buffer_data = self.lance_file.readBytes(page.buffer_offsets[0], page.buffer_sizes[0]) catch return TableError.InvalidMetadata;
            const decoder = PlainDecoder.init(buffer_data);
            const page_values = (switch (T) {
                i64 => decoder.readAllInt64(self.allocator),
                f64 => decoder.readAllFloat64(self.allocator),
                i32 => decoder.readAllInt32(self.allocator),
                f32 => decoder.readAllFloat32(self.allocator),
                else => @compileError("unsupported type"),
            }) catch return TableError.OutOfMemory;
            defer self.allocator.free(page_values);
            @memcpy(result[offset .. offset + page_values.len], page_values);
            offset += page_values.len;
        }

        return result;
    }

    pub fn readInt64Column(self: Self, col_idx: u32) TableError![]i64 {
        return self.readNumericColumn(i64, col_idx);
    }

    /// Gather specific row indices from an int64 column (avoids full column allocation)
    /// More efficient for JOINs where only a subset of rows are needed
    pub fn gatherInt64Column(self: Self, col_idx: u32, row_indices: []const usize, null_value: i64) TableError![]i64 {
        return self.gatherNumericColumn(i64, col_idx, row_indices, null_value);
    }

    /// Gather specific row indices from a float64 column
    pub fn gatherFloat64Column(self: Self, col_idx: u32, row_indices: []const usize, null_value: f64) TableError![]f64 {
        return self.gatherNumericColumn(f64, col_idx, row_indices, null_value);
    }

    /// Generic gather - reads full column, returns only specified indices
    /// More efficient than read + copy because it allocates only result size
    fn gatherNumericColumn(self: Self, comptime T: type, col_idx: u32, row_indices: []const usize, null_value: T) TableError![]T {
        const null_idx = std.math.maxInt(usize);

        // Read full column (Lance columnar format requires sequential read)
        const all_data = try self.readNumericColumn(T, col_idx);
        defer self.allocator.free(all_data);

        // Allocate result for gathered rows only
        const result = self.allocator.alloc(T, row_indices.len) catch return TableError.OutOfMemory;

        // Check if any null indices present (outer join case)
        var has_nulls = false;
        for (row_indices) |idx| {
            if (idx == null_idx) {
                has_nulls = true;
                break;
            }
        }

        // Gather values - branchless for inner join (no nulls)
        if (has_nulls) {
            for (row_indices, 0..) |idx, i| {
                result[i] = if (idx == null_idx) null_value else all_data[idx];
            }
        } else {
            // Fast path: no null check needed (inner join)
            // Use prefetch to hint CPU about upcoming memory accesses
            // Distance of 16 elements (~128 bytes for i64) allows L1 prefetch to complete
            const prefetch_distance: usize = 16;
            for (row_indices, 0..) |idx, i| {
                // Prefetch next few indices to reduce cache misses
                if (i + prefetch_distance < row_indices.len) {
                    const next_idx = row_indices[i + prefetch_distance];
                    @prefetch(@as([*]const T, @ptrCast(&all_data[next_idx])), .{
                        .rw = .read,
                        .locality = 0, // Non-temporal - data won't be reused soon
                        .cache = .data,
                    });
                }
                result[i] = all_data[idx];
            }
        }

        return result;
    }

    pub fn readInt64ColumnByName(self: Self, name: []const u8) TableError![]i64 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readInt64Column(@intCast(idx));
    }

    pub fn readFloat64Column(self: Self, col_idx: u32) TableError![]f64 {
        return self.readNumericColumn(f64, col_idx);
    }

    pub fn readFloat64ColumnByName(self: Self, name: []const u8) TableError![]f64 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readFloat64Column(@intCast(idx));
    }

    pub fn readInt32Column(self: Self, col_idx: u32) TableError![]i32 {
        return self.readNumericColumn(i32, col_idx);
    }

    pub fn readInt32ColumnByName(self: Self, name: []const u8) TableError![]i32 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readInt32Column(@intCast(idx));
    }

    pub fn readFloat32Column(self: Self, col_idx: u32) TableError![]f32 {
        return self.readNumericColumn(f32, col_idx);
    }

    pub fn readFloat32ColumnByName(self: Self, name: []const u8) TableError![]f32 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readFloat32Column(@intCast(idx));
    }

    /// Read all boolean values from a column (reads ALL pages).
    pub fn readBoolColumn(self: Self, col_idx: u32) TableError![]bool {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;

        // Calculate total rows across all pages
        var total_rows: usize = 0;
        for (col_meta.pages) |page| {
            total_rows += @intCast(page.length);
        }

        // Allocate result buffer for all pages
        var result = self.allocator.alloc(bool, total_rows) catch return TableError.OutOfMemory;
        errdefer self.allocator.free(result);

        // Read each page's buffer and decode
        var offset: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len == 0 or page.buffer_sizes.len == 0) continue;

            const buffer_offset = page.buffer_offsets[0];
            const buffer_size = page.buffer_sizes[0];
            const page_rows: usize = @intCast(page.length);

            const buffer_data = self.lance_file.readBytes(buffer_offset, buffer_size) catch {
                return TableError.InvalidMetadata;
            };

            const decoder = PlainDecoder.init(buffer_data);
            const page_values = decoder.readAllBool(self.allocator, page_rows) catch return TableError.OutOfMemory;
            defer self.allocator.free(page_values);

            @memcpy(result[offset .. offset + page_values.len], page_values);
            offset += page_values.len;
        }

        return result;
    }

    /// Read all boolean values from a column by name.
    pub fn readBoolColumnByName(self: Self, name: []const u8) TableError![]bool {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readBoolColumn(@intCast(idx));
    }

    // ========================================================================
    // Read At Indices - Phase 2 API for efficient sparse reads
    // ========================================================================
    // These methods read only specific row indices, enabling efficient
    // column projection after WHERE filtering.
    //
    // Current implementation: reads full column then filters (fallback)
    // Future optimization: direct byte-range reads for fixed-size types

    /// Read int64 values at specific row indices.
    /// Generic helper to filter array by indices with bounds checking
    fn filterByIndicesChecked(self: Self, comptime T: type, all_data: []const T, indices: []const u32) TableError![]T {
        const result = self.allocator.alloc(T, indices.len) catch return TableError.OutOfMemory;
        errdefer self.allocator.free(result);
        for (indices, 0..) |idx, i| {
            if (idx >= all_data.len) return TableError.IndexOutOfBounds;
            result[i] = all_data[idx];
        }
        return result;
    }

    pub fn readInt64AtIndices(self: Self, col_idx: u32, indices: []const u32) TableError![]i64 {
        const all_data = try self.readInt64Column(col_idx);
        defer self.allocator.free(all_data);
        return self.filterByIndicesChecked(i64, all_data, indices);
    }

    /// Read float64 values at specific row indices.
    pub fn readFloat64AtIndices(self: Self, col_idx: u32, indices: []const u32) TableError![]f64 {
        const all_data = try self.readFloat64Column(col_idx);
        defer self.allocator.free(all_data);
        return self.filterByIndicesChecked(f64, all_data, indices);
    }

    /// Read int32 values at specific row indices.
    pub fn readInt32AtIndices(self: Self, col_idx: u32, indices: []const u32) TableError![]i32 {
        const all_data = try self.readInt32Column(col_idx);
        defer self.allocator.free(all_data);
        return self.filterByIndicesChecked(i32, all_data, indices);
    }

    /// Read float32 values at specific row indices.
    pub fn readFloat32AtIndices(self: Self, col_idx: u32, indices: []const u32) TableError![]f32 {
        const all_data = try self.readFloat32Column(col_idx);
        defer self.allocator.free(all_data);
        return self.filterByIndicesChecked(f32, all_data, indices);
    }

    /// Read boolean values at specific row indices.
    pub fn readBoolAtIndices(self: Self, col_idx: u32, indices: []const u32) TableError![]bool {
        const all_data = try self.readBoolColumn(col_idx);
        defer self.allocator.free(all_data);
        return self.filterByIndicesChecked(bool, all_data, indices);
    }

    /// Read string values at specific row indices.
    /// Caller must free both the returned slice and each string.
    pub fn readStringAtIndices(self: Self, col_idx: u32, indices: []const u32) TableError![][]const u8 {
        const all_data = try self.readStringColumn(col_idx);
        defer {
            for (all_data) |s| self.allocator.free(s);
            self.allocator.free(all_data);
        }

        const result = self.allocator.alloc([]const u8, indices.len) catch return TableError.OutOfMemory;
        errdefer self.allocator.free(result);

        for (indices, 0..) |idx, i| {
            if (idx >= all_data.len) return TableError.IndexOutOfBounds;
            result[i] = self.allocator.dupe(u8, all_data[idx]) catch return TableError.OutOfMemory;
        }
        return result;
    }

    /// Read raw column buffer (first page, first buffer).
    /// For multi-page support, use typed column readers (readInt64Column, etc.).
    pub fn getColumnBuffer(self: Self, col_idx: u32) TableError![]const u8 {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) {
            return TableError.InvalidMetadata;
        }

        const page = col_meta.pages[0];
        if (page.buffer_offsets.len == 0) {
            return TableError.InvalidMetadata;
        }

        const buffer_offset = page.buffer_offsets[0];
        const buffer_size = page.buffer_sizes[0];

        return self.lance_file.readBytes(buffer_offset, buffer_size) catch {
            return TableError.InvalidMetadata;
        };
    }

    /// Get multiple buffers for a column (needed for variable-length types like strings).
    /// Returns a slice of buffers corresponding to the requested buffer indices.
    fn getColumnBuffers(self: Self, col_idx: u32, buffer_indices: []const usize) TableError![][]const u8 {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;
        const page = col_meta.pages[0];

        var buffers = try self.allocator.alloc([]const u8, buffer_indices.len);
        errdefer self.allocator.free(buffers);

        for (buffer_indices, 0..) |buf_idx, i| {
            if (buf_idx >= page.buffer_offsets.len) return TableError.InvalidBufferIndex;

            const buffer_offset = page.buffer_offsets[buf_idx];
            const buffer_size = page.buffer_sizes[buf_idx];

            buffers[i] = self.lance_file.readBytes(buffer_offset, buffer_size) catch {
                return TableError.InvalidMetadata;
            };
        }

        return buffers;
    }

    /// Get the number of rows in a specific column.
    /// Reads the column metadata and sums the length across all pages.
    fn numRows(self: Self, col_idx: u32) TableError!usize {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        // Sum length across all pages
        var total_rows: usize = 0;
        for (col_meta.pages) |page| {
            total_rows += @intCast(page.length);
        }

        return total_rows;
    }

    /// Read a string column by index (reads ALL pages).
    /// Returns a slice of allocated strings (UTF-8 byte slices).
    /// Caller must free each string AND the slice itself using the same allocator.
    pub fn readStringColumn(self: Self, col_idx: u32) TableError![][]const u8 {
        // Get column metadata
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;

        // Calculate total rows across all pages
        var total_rows: usize = 0;
        for (col_meta.pages) |page| {
            total_rows += @intCast(page.length);
        }

        // Allocate result buffer for all pages
        var owned_strings = self.allocator.alloc([]const u8, total_rows) catch {
            return TableError.OutOfMemory;
        };
        errdefer {
            for (owned_strings) |str| {
                if (str.len > 0) self.allocator.free(str);
            }
            self.allocator.free(owned_strings);
        }

        // Read each page
        var result_offset: usize = 0;
        for (col_meta.pages) |page| {
            // Lance stores string columns with TWO separate buffers per page:
            // - Buffer 0: offsets array (uint32 or uint64, marking END positions)
            // - Buffer 1: string data (concatenated UTF-8 bytes)
            // Dictionary-encoded columns have 3+ buffers - not yet supported
            if (page.buffer_offsets.len < 2) {
                return TableError.InvalidMetadata;
            }
            if (page.buffer_offsets.len > 2) {
                // Dictionary encoding not yet supported
                return TableError.DictionaryEncodingNotSupported;
            }

            // Buffer 0 = offsets array
            const offsets_offset = page.buffer_offsets[0];
            const offsets_size = page.buffer_sizes[0];
            const offsets_buffer = self.lance_file.readBytes(offsets_offset, offsets_size) catch {
                return TableError.InvalidMetadata;
            };

            // Buffer 1 = string data
            const data_offset = page.buffer_offsets[1];
            const data_size = page.buffer_sizes[1];
            const data_buffer = self.lance_file.readBytes(data_offset, data_size) catch {
                return TableError.InvalidMetadata;
            };

            // Decode strings (returns slices into data_buffer, not owned copies)
            const string_slices = PlainDecoder.readAllStrings(offsets_buffer, data_buffer, self.allocator) catch {
                return TableError.InvalidMetadata;
            };
            defer self.allocator.free(string_slices);

            // Copy each string into owned memory
            for (string_slices, 0..) |slice, i| {
                const copy = self.allocator.alloc(u8, slice.len) catch {
                    // Mark how many we successfully allocated for errdefer
                    owned_strings = owned_strings[0 .. result_offset + i];
                    return TableError.OutOfMemory;
                };
                @memcpy(copy, slice);
                owned_strings[result_offset + i] = copy;
            }
            result_offset += string_slices.len;
        }

        return owned_strings;
    }

    /// Read a string column by name.
    pub fn readStringColumnByName(self: Self, name: []const u8) TableError![][]const u8 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.readStringColumn(@intCast(idx));
    }

    /// String column buffers for zero-copy Arrow export.
    pub const StringBuffers = struct {
        offsets: []const u8, // Raw Lance offsets buffer (int32 end positions)
        data: []const u8, // Raw string data buffer
    };

    /// Get raw string column buffers for zero-copy Arrow export (reads ALL pages).
    /// Returns merged offsets and data buffers.
    /// Caller must free both buffers using the table's allocator.
    pub fn getStringColumnBuffers(self: Self, col_idx: u32) TableError!StringBuffers {
        const col_meta_bytes = self.lance_file.getColumnMetadataBytes(col_idx) catch {
            return TableError.ColumnOutOfBounds;
        };

        var col_meta = ColumnMetadata.parse(self.allocator, col_meta_bytes) catch {
            return TableError.InvalidMetadata;
        };
        defer col_meta.deinit(self.allocator);

        if (col_meta.pages.len == 0) return TableError.NoPages;

        // Single page - return directly (zero-copy)
        if (col_meta.pages.len == 1) {
            const page = col_meta.pages[0];
            if (page.buffer_offsets.len < 2) return TableError.InvalidMetadata;

            const offsets_buffer = self.lance_file.readBytes(page.buffer_offsets[0], page.buffer_sizes[0]) catch {
                return TableError.InvalidMetadata;
            };
            const data_buffer = self.lance_file.readBytes(page.buffer_offsets[1], page.buffer_sizes[1]) catch {
                return TableError.InvalidMetadata;
            };

            return StringBuffers{
                .offsets = offsets_buffer,
                .data = data_buffer,
            };
        }

        // Multiple pages - need to merge offsets and data buffers
        // Calculate total sizes
        var total_offsets_size: usize = 0;
        var total_data_size: usize = 0;
        for (col_meta.pages) |page| {
            if (page.buffer_sizes.len >= 2) {
                total_offsets_size += page.buffer_sizes[0];
                total_data_size += page.buffer_sizes[1];
            }
        }

        // Allocate merged buffers
        const merged_offsets = self.allocator.alloc(u8, total_offsets_size) catch return TableError.OutOfMemory;
        errdefer self.allocator.free(merged_offsets);

        const merged_data = self.allocator.alloc(u8, total_data_size) catch return TableError.OutOfMemory;
        errdefer self.allocator.free(merged_data);

        // Copy data from each page, adjusting offsets
        var offsets_pos: usize = 0;
        var data_pos: usize = 0;
        var data_offset_adjustment: u32 = 0;

        for (col_meta.pages) |page| {
            if (page.buffer_offsets.len < 2) continue;

            // Read page buffers
            const page_offsets = self.lance_file.readBytes(page.buffer_offsets[0], page.buffer_sizes[0]) catch {
                return TableError.InvalidMetadata;
            };
            const page_data = self.lance_file.readBytes(page.buffer_offsets[1], page.buffer_sizes[1]) catch {
                return TableError.InvalidMetadata;
            };

            // Copy and adjust offsets (Lance uses i32 end-offsets)
            const num_offsets = page_offsets.len / @sizeOf(i32);
            const src_offsets = @as([*]const i32, @ptrCast(@alignCast(page_offsets.ptr)))[0..num_offsets];
            const dst_offsets = @as([*]i32, @ptrCast(@alignCast(merged_offsets.ptr + offsets_pos)))[0..num_offsets];

            for (src_offsets, 0..) |offset, i| {
                dst_offsets[i] = offset + @as(i32, @intCast(data_offset_adjustment));
            }

            // Copy data
            @memcpy(merged_data[data_pos .. data_pos + page_data.len], page_data);

            offsets_pos += page_offsets.len;
            data_pos += page_data.len;
            data_offset_adjustment = @intCast(data_pos);
        }

        return StringBuffers{
            .offsets = merged_offsets,
            .data = merged_data,
        };
    }

    // ========================================================================
    // Compute Operations (using SIMD with auto threshold dispatch)
    // ========================================================================

    /// Compute L2 norm (Euclidean length) of a float64 column.
    /// Uses SIMD with automatic threshold-based dispatch.
    pub fn computeL2Norm(self: Self, col_idx: u32) TableError!f64 {
        const data = try self.readFloat64Column(col_idx);
        defer self.allocator.free(data);
        return simd.l2Norm(data);
    }

    /// Compute L2 norm by column name.
    pub fn computeL2NormByName(self: Self, name: []const u8) TableError!f64 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.computeL2Norm(@intCast(idx));
    }

    /// Compute sum of a float64 column.
    /// Uses SIMD with automatic threshold-based dispatch.
    pub fn computeSum(self: Self, col_idx: u32) TableError!f64 {
        const data = try self.readFloat64Column(col_idx);
        defer self.allocator.free(data);
        return simd.sum(data);
    }

    /// Compute sum by column name.
    pub fn computeSumByName(self: Self, name: []const u8) TableError!f64 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.computeSum(@intCast(idx));
    }

    /// Count rows where column value > threshold.
    /// Uses SIMD with automatic threshold-based dispatch.
    pub fn countGreaterThan(self: Self, col_idx: u32, threshold: f64) TableError!u64 {
        const data = try self.readFloat64Column(col_idx);
        defer self.allocator.free(data);
        return simd.countGreaterThan(data, threshold);
    }

    /// Count rows where column value > threshold, by column name.
    pub fn countGreaterThanByName(self: Self, name: []const u8, threshold: f64) TableError!u64 {
        const idx = self.columnIndex(name) orelse return TableError.ColumnNotFound;
        return self.countGreaterThan(@intCast(idx), threshold);
    }

    /// Compute dot product of two float64 columns.
    /// Uses SIMD with automatic threshold-based dispatch.
    pub fn computeDotProduct(self: Self, col_a: u32, col_b: u32) TableError!f64 {
        const data_a = try self.readFloat64Column(col_a);
        defer self.allocator.free(data_a);
        const data_b = try self.readFloat64Column(col_b);
        defer self.allocator.free(data_b);
        return simd.dotProduct(data_a, data_b);
    }

    /// Read validity bitmap for a nullable column
    /// Returns null if column has no validity bitmap (non-nullable or all values valid)
    /// The bitmap uses Arrow format: bit 1 = valid, bit 0 = null
    /// NOTE: This in-memory Table doesn't support validity bitmaps yet - returns null
    pub fn readValidityBitmap(self: Self, col_idx: u32) TableError!?[]u8 {
        _ = self;
        _ = col_idx;
        // In-memory Table uses LanceFile which doesn't have validity bitmap support yet
        // Return null to indicate all values are valid
        return null;
    }

    /// Batch dot product: compute dot products of f32 embeddings against a query vector.
    /// Returns array of scores (one per row).
    /// Caller must free the returned slice.
    pub fn batchDotProduct(self: Self, col_idx: u32, query: []const f64, dim: usize) TableError![]f64 {
        const embeddings = try self.readFloat32Column(col_idx);
        defer self.allocator.free(embeddings);

        const num_rows = embeddings.len / dim;
        const out = self.allocator.alloc(f64, num_rows) catch return TableError.OutOfMemory;
        errdefer self.allocator.free(out);

        simd.batchDotProductF32(embeddings, query, dim, out);
        return out;
    }
};

// ============================================================================
// LazyTable - File-path based API using column-first I/O
// ============================================================================

/// Lazy table reader - only reads columns on demand, directly from file
///
/// Benefits over Table:
/// - No need to load entire file into memory first
/// - Column-first I/O - only reads the bytes for requested columns
/// - 10-100x I/O reduction for queries that only need 1-2 columns
pub const LazyTable = struct {
    allocator: std.mem.Allocator,
    file_reader: *FileReader, // Heap-allocated to ensure stable address
    lazy_file: LazyLanceFile,
    schema: ?Schema,
    schema_bytes: ?[]const u8,

    const Self = @This();

    /// Open a lazy table from a file path
    pub fn initFromPath(allocator: std.mem.Allocator, path: []const u8) TableError!Self {
        // Allocate FileReader on heap so its address stays stable
        const file_reader = allocator.create(FileReader) catch {
            return TableError.OutOfMemory;
        };
        errdefer allocator.destroy(file_reader);

        file_reader.* = FileReader.open(path) catch {
            return TableError.ReadError;
        };
        errdefer file_reader.close();

        var lazy_file = LazyLanceFile.init(allocator, file_reader.reader()) catch |err| {
            return switch (err) {
                error.FileTooSmall => TableError.FileTooSmall,
                error.InvalidMagic => TableError.InvalidMagic,
                error.UnsupportedVersion => TableError.UnsupportedVersion,
                error.InvalidMetadata => TableError.InvalidMetadata,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
        errdefer lazy_file.deinit();

        // Parse schema from global buffer 0 (lazy read)
        var schema: ?Schema = null;
        var schema_bytes: ?[]const u8 = null;
        if (lazy_file.getSchemaBytes() catch null) |bytes| {
            schema_bytes = bytes;
            schema = Schema.parse(allocator, bytes) catch null;
        }

        return Self{
            .allocator = allocator,
            .file_reader = file_reader,
            .lazy_file = lazy_file,
            .schema = schema,
            .schema_bytes = schema_bytes,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.schema) |*s| s.deinit();
        if (self.schema_bytes) |bytes| self.allocator.free(bytes);
        self.lazy_file.deinit();
        self.file_reader.close();
        self.allocator.destroy(self.file_reader);
    }

    /// Get the number of columns
    pub fn numColumns(self: Self) u32 {
        return self.lazy_file.numColumns();
    }

    /// Get the schema
    pub fn getSchema(self: Self) ?Schema {
        return self.schema;
    }

    /// Get column index by name
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        const schema = self.schema orelse return null;
        return schema.fieldIndex(name);
    }

    /// Get physical column ID by name
    pub fn physicalColumnId(self: Self, name: []const u8) ?u32 {
        const schema = self.schema orelse return null;
        return schema.physicalColumnId(name);
    }

    /// Get field by physical ID
    pub fn getFieldById(self: Self, id: u32) ?Field {
        const schema = self.schema orelse return null;
        for (schema.fields) |field| {
            if (field.id == id) return field;
        }
        return null;
    }

    /// Get column names from schema
    pub fn columnNames(self: Self) TableError![][]const u8 {
        const schema = self.schema orelse return TableError.NoSchema;
        return schema.columnNames(self.allocator) catch return TableError.OutOfMemory;
    }

    /// Get row count by reading a column's metadata
    pub fn rowCount(self: *Self, col_idx: u32) TableError!u32 {
        // Read int64 column and return its length
        // This is a bit wasteful but ensures consistency
        const data = try self.readInt64Column(col_idx);
        defer self.allocator.free(data);
        return @intCast(data.len);
    }

    /// Read int64 column by physical ID
    pub fn readInt64Column(self: *Self, col_idx: u32) TableError![]i64 {
        return self.lazy_file.readInt64Column(col_idx) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read int32 column by physical ID
    pub fn readInt32Column(self: *Self, col_idx: u32) TableError![]i32 {
        return self.lazy_file.readInt32Column(col_idx) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read float64 column by physical ID
    pub fn readFloat64Column(self: *Self, col_idx: u32) TableError![]f64 {
        return self.lazy_file.readFloat64Column(col_idx) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read float32 column by physical ID
    pub fn readFloat32Column(self: *Self, col_idx: u32) TableError![]f32 {
        return self.lazy_file.readFloat32Column(col_idx) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read bool column by physical ID
    pub fn readBoolColumn(self: *Self, col_idx: u32) TableError![]bool {
        return self.lazy_file.readBoolColumn(col_idx) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read string column by physical ID
    pub fn readStringColumn(self: *Self, col_idx: u32) TableError![][]const u8 {
        return self.lazy_file.readStringColumn(col_idx) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read validity bitmap for a nullable column
    /// Returns null if column has no validity bitmap (non-nullable or all values valid)
    /// The bitmap uses Arrow format: bit 1 = valid, bit 0 = null
    pub fn readValidityBitmap(self: *Self, col_idx: u32) TableError!?[]u8 {
        return self.lazy_file.readValidityBitmap(col_idx) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    // ========================================================================
    // Selective Row Reading (Late Materialization Support)
    // ========================================================================

    /// Read int64 values at specific row indices.
    /// TRUE selective read - only fetches bytes for requested rows.
    /// This is the key method for late materialization.
    pub fn readInt64ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) TableError![]i64 {
        return self.lazy_file.readInt64ColumnAtIndices(col_idx, row_indices) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read float64 values at specific row indices.
    /// TRUE selective read - only fetches bytes for requested rows.
    pub fn readFloat64ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) TableError![]f64 {
        return self.lazy_file.readFloat64ColumnAtIndices(col_idx, row_indices) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read int32 values at specific row indices.
    /// TRUE selective read - only fetches bytes for requested rows.
    pub fn readInt32ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) TableError![]i32 {
        return self.lazy_file.readInt32ColumnAtIndices(col_idx, row_indices) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }

    /// Read float32 values at specific row indices.
    /// TRUE selective read - only fetches bytes for requested rows.
    pub fn readFloat32ColumnAtIndices(self: *Self, col_idx: u32, row_indices: []const u32) TableError![]f32 {
        return self.lazy_file.readFloat32ColumnAtIndices(col_idx, row_indices) catch |err| {
            return switch (err) {
                error.ColumnOutOfBounds => TableError.ColumnOutOfBounds,
                error.NoPages => TableError.NoPages,
                error.OutOfMemory => TableError.OutOfMemory,
                else => TableError.ReadError,
            };
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "table error enum" {
    // Just verify the error set compiles
    const err: TableError = TableError.NoSchema;
    try std.testing.expect(err == TableError.NoSchema);
}

test "table: read int64 column from lance file" {
    const allocator = std.testing.allocator;

    // Open the data file from simple_int64.lance/data/
    var dir = std.fs.cwd().openDir("tests/fixtures/simple_int64.lance/data", .{ .iterate = true }) catch return error.SkipZigTest;
    defer dir.close();

    // Get the first .lance file
    var iter = dir.iterate();
    const entry = iter.next() catch return error.SkipZigTest;
    const filename = (entry orelse return error.SkipZigTest).name;

    // Read file
    const file = dir.openFile(filename, .{}) catch return error.SkipZigTest;
    defer file.close();

    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    // Initialize table
    var table = Table.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Read values (column 0 = id)
    const values = table.readInt64Column(0) catch return error.SkipZigTest;
    defer allocator.free(values);

    // simple_int64.lance contains 5 rows: 1, 2, 3, 4, 5
    try std.testing.expectEqual(@as(usize, 5), values.len);
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 5), values[4]);
}

test "table: read float64 column from lance file" {
    const allocator = std.testing.allocator;

    // Open the data file from simple_float64.lance/data/
    var dir = std.fs.cwd().openDir("tests/fixtures/simple_float64.lance/data", .{ .iterate = true }) catch return error.SkipZigTest;
    defer dir.close();

    var iter = dir.iterate();
    const entry = iter.next() catch return error.SkipZigTest;
    const filename = (entry orelse return error.SkipZigTest).name;

    const file = dir.openFile(filename, .{}) catch return error.SkipZigTest;
    defer file.close();

    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = Table.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    const values = table.readFloat64Column(0) catch return error.SkipZigTest;
    defer allocator.free(values);

    // simple_float64.lance contains 5 rows: 1.1, 2.2, 3.3, 4.4, 5.5
    try std.testing.expectEqual(@as(usize, 5), values.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), values[4], 0.01);
}

test "table: schema and column names" {
    const allocator = std.testing.allocator;

    // Open the data file from simple_int64.lance/data/
    var dir = std.fs.cwd().openDir("tests/fixtures/simple_int64.lance/data", .{ .iterate = true }) catch return error.SkipZigTest;
    defer dir.close();

    var iter = dir.iterate();
    const entry = iter.next() catch return error.SkipZigTest;
    const filename = (entry orelse return error.SkipZigTest).name;

    const file = dir.openFile(filename, .{}) catch return error.SkipZigTest;
    defer file.close();

    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = Table.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Get column names
    const names = table.columnNames() catch return error.SkipZigTest;
    defer allocator.free(names);

    try std.testing.expectEqual(@as(usize, 1), names.len);
    try std.testing.expectEqualStrings("id", names[0]);

    // Test column index lookup
    try std.testing.expectEqual(@as(?usize, 0), table.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, null), table.columnIndex("nonexistent"));
}

test "table: physical column id mapping" {
    const allocator = std.testing.allocator;

    // Open the data file from simple_int64.lance/data/
    var dir = std.fs.cwd().openDir("tests/fixtures/simple_int64.lance/data", .{ .iterate = true }) catch return error.SkipZigTest;
    defer dir.close();

    var iter = dir.iterate();
    const entry = iter.next() catch return error.SkipZigTest;
    const filename = (entry orelse return error.SkipZigTest).name;

    const file = dir.openFile(filename, .{}) catch return error.SkipZigTest;
    defer file.close();

    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = Table.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Test physical column ID
    const physical_id = table.physicalColumnId("id");
    try std.testing.expect(physical_id != null);

    // Nonexistent column should return null
    try std.testing.expectEqual(@as(?u32, null), table.physicalColumnId("nonexistent"));
}
