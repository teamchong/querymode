//! Page Row Index - Maps global row indices to byte offsets for selective reads.
//!
//! This module enables Late Materialization by providing efficient row-to-byte-offset
//! mapping, allowing selective reads of specific rows without loading entire columns.
//!
//! For fixed-size types (int64, float64, etc.), the byte offset is computed as:
//!   byte_offset = page.buffer_offsets[0] + local_row * sizeof(T)
//!
//! This enables batched HTTP range requests for cloud-hosted Lance files.

const std = @import("std");
const proto = @import("lanceql.proto");

const Page = proto.Page;
const ColumnMetadata = proto.ColumnMetadata;

/// Location of a single row within a page
pub const RowLocation = struct {
    /// Index of the page containing this row
    page_idx: u32,
    /// Local row index within the page (0-based)
    local_row: u32,
    /// Absolute byte offset in the file
    byte_offset: u64,
    /// Size of the value in bytes
    value_size: u32,
};

/// A contiguous byte range for batch reading
pub const ByteRange = struct {
    /// Start offset in file
    start: u64,
    /// End offset (exclusive) in file
    end: u64,
    /// First global row index covered by this range
    first_row: u32,
    /// Number of rows in this range
    row_count: u32,

    /// Get the byte length of this range
    pub fn len(self: ByteRange) u64 {
        return self.end - self.start;
    }
};

/// Index structure for mapping row indices to byte offsets within a column.
///
/// Enables O(log n) lookup of any row's byte location across multiple pages.
/// Used for selective reads in late materialization.
pub const PageRowIndex = struct {
    pages: []const Page,
    /// Cumulative row counts: cumulative_rows[i] = total rows in pages 0..i
    /// Used for binary search to find page containing a given global row
    cumulative_rows: []u64,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Build a PageRowIndex from column metadata.
    /// Caller owns the returned index and must call deinit().
    pub fn init(allocator: std.mem.Allocator, col_meta: *const ColumnMetadata) !Self {
        const pages = col_meta.pages;
        if (pages.len == 0) {
            return Self{
                .pages = pages,
                .cumulative_rows = &[_]u64{},
                .allocator = allocator,
            };
        }

        // Build cumulative row counts for binary search
        const cumulative = try allocator.alloc(u64, pages.len);
        errdefer allocator.free(cumulative);

        var total: u64 = 0;
        for (pages, 0..) |page, i| {
            total += page.length;
            cumulative[i] = total;
        }

        return Self{
            .pages = pages,
            .cumulative_rows = cumulative,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.cumulative_rows.len > 0) {
            self.allocator.free(self.cumulative_rows);
        }
    }

    /// Get total row count across all pages
    pub fn totalRows(self: Self) u64 {
        if (self.cumulative_rows.len == 0) return 0;
        return self.cumulative_rows[self.cumulative_rows.len - 1];
    }

    /// Find which page contains a given global row index.
    /// Returns (page_idx, local_row_within_page).
    pub fn findPage(self: Self, global_row: u64) ?struct { page_idx: u32, local_row: u32 } {
        if (self.pages.len == 0 or global_row >= self.totalRows()) {
            return null;
        }

        // Binary search for the page containing this row
        var low: usize = 0;
        var high: usize = self.cumulative_rows.len;

        while (low < high) {
            const mid = low + (high - low) / 2;
            if (self.cumulative_rows[mid] <= global_row) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        const page_idx: u32 = @intCast(low);

        // Calculate local row within page
        const page_start_row = if (page_idx == 0) 0 else self.cumulative_rows[page_idx - 1];
        const local_row: u32 = @intCast(global_row - page_start_row);

        return .{ .page_idx = page_idx, .local_row = local_row };
    }

    /// Get the byte offset for a single row of a fixed-size type.
    ///
    /// Parameters:
    ///   - global_row: Global row index (0-based across all pages)
    ///   - value_size: Size of each value in bytes (e.g., 8 for int64)
    ///   - buffer_idx: Which buffer within the page to read from (usually 0)
    ///
    /// Returns the RowLocation with absolute byte offset, or null if out of bounds.
    pub fn getFixedRowOffset(self: Self, global_row: u64, value_size: usize, buffer_idx: usize) ?RowLocation {
        const loc = self.findPage(global_row) orelse return null;
        const page = self.pages[loc.page_idx];

        if (buffer_idx >= page.buffer_offsets.len) {
            return null;
        }

        const base_offset = page.buffer_offsets[buffer_idx];
        const byte_offset = base_offset + @as(u64, loc.local_row) * @as(u64, @intCast(value_size));

        return RowLocation{
            .page_idx = loc.page_idx,
            .local_row = loc.local_row,
            .byte_offset = byte_offset,
            .value_size = @intCast(value_size),
        };
    }

    /// Compute byte ranges for multiple row indices, coalescing nearby ranges.
    ///
    /// This is the key function for efficient batch HTTP reads. It:
    /// 1. Maps each row index to its byte offset
    /// 2. Sorts by byte offset
    /// 3. Coalesces ranges within gap_threshold bytes of each other
    ///
    /// Parameters:
    ///   - rows: Global row indices to read (need not be sorted)
    ///   - value_size: Size of each value in bytes
    ///   - gap_threshold: Merge ranges if gap between them is <= this (e.g., 4096)
    ///   - buffer_idx: Which buffer within pages to read from (usually 0)
    ///
    /// Returns slice of coalesced ByteRanges. Caller owns the memory.
    pub fn getByteRanges(
        self: Self,
        rows: []const u32,
        value_size: usize,
        gap_threshold: u64,
        buffer_idx: usize,
    ) ![]ByteRange {
        if (rows.len == 0) {
            return self.allocator.alloc(ByteRange, 0);
        }

        // Step 1: Map rows to byte offsets with row metadata
        const RowOffset = struct {
            global_row: u32,
            byte_offset: u64,
        };

        var offsets = try self.allocator.alloc(RowOffset, rows.len);
        defer self.allocator.free(offsets);

        var valid_count: usize = 0;
        for (rows) |row| {
            if (self.getFixedRowOffset(row, value_size, buffer_idx)) |loc| {
                offsets[valid_count] = .{
                    .global_row = @intCast(row),
                    .byte_offset = loc.byte_offset,
                };
                valid_count += 1;
            }
        }

        if (valid_count == 0) {
            return self.allocator.alloc(ByteRange, 0);
        }

        // Step 2: Sort by byte offset
        const valid_offsets = offsets[0..valid_count];
        std.mem.sort(RowOffset, valid_offsets, {}, struct {
            fn lessThan(_: void, a: RowOffset, b: RowOffset) bool {
                return a.byte_offset < b.byte_offset;
            }
        }.lessThan);

        // Step 3: Coalesce adjacent ranges
        var ranges = std.ArrayList(ByteRange).init(self.allocator);
        errdefer ranges.deinit();

        var current_start = valid_offsets[0].byte_offset;
        var current_end = valid_offsets[0].byte_offset + @as(u64, @intCast(value_size));
        var current_first_row = valid_offsets[0].global_row;
        var current_row_count: u32 = 1;

        for (valid_offsets[1..]) |off| {
            const off_end = off.byte_offset + @as(u64, @intCast(value_size));

            // Check if this offset can be merged with current range
            if (off.byte_offset <= current_end + gap_threshold) {
                // Extend current range
                current_end = @max(current_end, off_end);
                current_row_count += 1;
            } else {
                // Flush current range and start new one
                try ranges.append(ByteRange{
                    .start = current_start,
                    .end = current_end,
                    .first_row = current_first_row,
                    .row_count = current_row_count,
                });

                current_start = off.byte_offset;
                current_end = off_end;
                current_first_row = off.global_row;
                current_row_count = 1;
            }
        }

        // Flush final range
        try ranges.append(ByteRange{
            .start = current_start,
            .end = current_end,
            .first_row = current_first_row,
            .row_count = current_row_count,
        });

        return ranges.toOwnedSlice();
    }

    /// Get byte ranges for reading values at specific indices, with range merging.
    /// Specialized version for common case of reading int64/float64 values.
    pub fn getByteRangesForType(
        self: Self,
        comptime T: type,
        rows: []const u32,
        gap_threshold: u64,
    ) ![]ByteRange {
        return self.getByteRanges(rows, @sizeOf(T), gap_threshold, 0);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "PageRowIndex: empty metadata" {
    const allocator = std.testing.allocator;

    // Create empty ColumnMetadata
    var meta = ColumnMetadata{
        .encoding = .none,
        .pages = &[_]Page{},
        .buffer_offsets = &[_]u64{},
        .buffer_sizes = &[_]u64{},
    };

    var idx = try PageRowIndex.init(allocator, &meta);
    defer idx.deinit();

    try std.testing.expectEqual(@as(u64, 0), idx.totalRows());
    try std.testing.expectEqual(@as(?RowLocation, null), idx.getFixedRowOffset(0, 8, 0));
}

test "PageRowIndex: single page" {
    const allocator = std.testing.allocator;

    // Create a single page with 100 rows, int64 values starting at offset 1000
    const page = Page{
        .buffer_offsets = &[_]u64{1000},
        .buffer_sizes = &[_]u64{800}, // 100 * 8 bytes
        .length = 100,
        .encoding = .none,
        .priority = 0,
    };
    const pages = [_]Page{page};

    var meta = ColumnMetadata{
        .encoding = .none,
        .pages = &pages,
        .buffer_offsets = &[_]u64{},
        .buffer_sizes = &[_]u64{},
    };

    var idx = try PageRowIndex.init(allocator, &meta);
    defer idx.deinit();

    try std.testing.expectEqual(@as(u64, 100), idx.totalRows());

    // Row 0 should be at offset 1000
    const loc0 = idx.getFixedRowOffset(0, 8, 0).?;
    try std.testing.expectEqual(@as(u32, 0), loc0.page_idx);
    try std.testing.expectEqual(@as(u32, 0), loc0.local_row);
    try std.testing.expectEqual(@as(u64, 1000), loc0.byte_offset);

    // Row 50 should be at offset 1000 + 50*8 = 1400
    const loc50 = idx.getFixedRowOffset(50, 8, 0).?;
    try std.testing.expectEqual(@as(u32, 0), loc50.page_idx);
    try std.testing.expectEqual(@as(u32, 50), loc50.local_row);
    try std.testing.expectEqual(@as(u64, 1400), loc50.byte_offset);

    // Row 99 should be at offset 1000 + 99*8 = 1792
    const loc99 = idx.getFixedRowOffset(99, 8, 0).?;
    try std.testing.expectEqual(@as(u64, 1792), loc99.byte_offset);

    // Row 100 should be out of bounds
    try std.testing.expectEqual(@as(?RowLocation, null), idx.getFixedRowOffset(100, 8, 0));
}

test "PageRowIndex: multiple pages" {
    const allocator = std.testing.allocator;

    // Page 0: 100 rows starting at offset 0
    // Page 1: 200 rows starting at offset 10000
    const page0 = Page{
        .buffer_offsets = &[_]u64{0},
        .buffer_sizes = &[_]u64{800},
        .length = 100,
        .encoding = .none,
        .priority = 0,
    };
    const page1 = Page{
        .buffer_offsets = &[_]u64{10000},
        .buffer_sizes = &[_]u64{1600},
        .length = 200,
        .encoding = .none,
        .priority = 100,
    };
    const pages = [_]Page{ page0, page1 };

    var meta = ColumnMetadata{
        .encoding = .none,
        .pages = &pages,
        .buffer_offsets = &[_]u64{},
        .buffer_sizes = &[_]u64{},
    };

    var idx = try PageRowIndex.init(allocator, &meta);
    defer idx.deinit();

    try std.testing.expectEqual(@as(u64, 300), idx.totalRows());

    // Row 0 in page 0
    const loc0 = idx.getFixedRowOffset(0, 8, 0).?;
    try std.testing.expectEqual(@as(u32, 0), loc0.page_idx);
    try std.testing.expectEqual(@as(u64, 0), loc0.byte_offset);

    // Row 99 is last row in page 0
    const loc99 = idx.getFixedRowOffset(99, 8, 0).?;
    try std.testing.expectEqual(@as(u32, 0), loc99.page_idx);
    try std.testing.expectEqual(@as(u64, 792), loc99.byte_offset);

    // Row 100 is first row in page 1
    const loc100 = idx.getFixedRowOffset(100, 8, 0).?;
    try std.testing.expectEqual(@as(u32, 1), loc100.page_idx);
    try std.testing.expectEqual(@as(u32, 0), loc100.local_row);
    try std.testing.expectEqual(@as(u64, 10000), loc100.byte_offset);

    // Row 150 is row 50 in page 1
    const loc150 = idx.getFixedRowOffset(150, 8, 0).?;
    try std.testing.expectEqual(@as(u32, 1), loc150.page_idx);
    try std.testing.expectEqual(@as(u32, 50), loc150.local_row);
    try std.testing.expectEqual(@as(u64, 10400), loc150.byte_offset);
}

test "PageRowIndex: byte range coalescing" {
    const allocator = std.testing.allocator;

    // Single page with 10000 rows starting at offset 0
    const page = Page{
        .buffer_offsets = &[_]u64{0},
        .buffer_sizes = &[_]u64{80000},
        .length = 10000,
        .encoding = .none,
        .priority = 0,
    };
    const pages = [_]Page{page};

    var meta = ColumnMetadata{
        .encoding = .none,
        .pages = &pages,
        .buffer_offsets = &[_]u64{},
        .buffer_sizes = &[_]u64{},
    };

    var idx = try PageRowIndex.init(allocator, &meta);
    defer idx.deinit();

    // Test case from plan: rows [2, 105, 5000, 5001, 5002, 8000]
    // For int64 (8 bytes/row):
    //   Row 2:    offset 16
    //   Row 105:  offset 840
    //   Row 5000: offset 40000
    //   Row 5001: offset 40008
    //   Row 5002: offset 40016
    //   Row 8000: offset 64000
    const rows = [_]u32{ 2, 105, 5000, 5001, 5002, 8000 };

    // With gap_threshold=4KB, rows 5000-5002 should merge
    const ranges = try idx.getByteRangesForType(i64, &rows, 4096);
    defer allocator.free(ranges);

    // Should produce 4 ranges:
    // Range 1: 16-24 (row 2)
    // Range 2: 840-848 (row 105)
    // Range 3: 40000-40024 (rows 5000-5002 merged)
    // Range 4: 64000-64008 (row 8000)
    try std.testing.expectEqual(@as(usize, 4), ranges.len);

    try std.testing.expectEqual(@as(u64, 16), ranges[0].start);
    try std.testing.expectEqual(@as(u64, 24), ranges[0].end);

    try std.testing.expectEqual(@as(u64, 840), ranges[1].start);
    try std.testing.expectEqual(@as(u64, 848), ranges[1].end);

    try std.testing.expectEqual(@as(u64, 40000), ranges[2].start);
    try std.testing.expectEqual(@as(u64, 40024), ranges[2].end);
    try std.testing.expectEqual(@as(u32, 3), ranges[2].row_count);

    try std.testing.expectEqual(@as(u64, 64000), ranges[3].start);
    try std.testing.expectEqual(@as(u64, 64008), ranges[3].end);
}

test "PageRowIndex: empty row list" {
    const allocator = std.testing.allocator;

    const page = Page{
        .buffer_offsets = &[_]u64{0},
        .buffer_sizes = &[_]u64{800},
        .length = 100,
        .encoding = .none,
        .priority = 0,
    };
    const pages = [_]Page{page};

    var meta = ColumnMetadata{
        .encoding = .none,
        .pages = &pages,
        .buffer_offsets = &[_]u64{},
        .buffer_sizes = &[_]u64{},
    };

    var idx = try PageRowIndex.init(allocator, &meta);
    defer idx.deinit();

    const ranges = try idx.getByteRangesForType(i64, &[_]u32{}, 4096);
    defer allocator.free(ranges);

    try std.testing.expectEqual(@as(usize, 0), ranges.len);
}
