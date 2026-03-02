//! Batch Reader for Late Materialization
//!
//! This module provides efficient batch reading capabilities for selective column reads.
//! It integrates with PageRowIndex to convert row indices to byte ranges, then uses
//! the Reader's batch_read capability (or fallback sequential reads) to fetch data.
//!
//! Key features:
//! - Converts row indices to coalesced byte ranges
//! - Supports HTTP multi-range requests when available
//! - Falls back to sequential reads for non-HTTP readers
//! - Memory-efficient: allocates only what's needed

const std = @import("std");
const reader_mod = @import("reader.zig");
const format = @import("lanceql.format");

const Reader = reader_mod.Reader;
const ReadError = reader_mod.ReadError;
const ByteRange = reader_mod.ByteRange;
const BatchReadResult = reader_mod.BatchReadResult;
const PageRowIndex = format.PageRowIndex;
const FormatByteRange = format.ByteRange;

/// Default gap threshold for range coalescing (4KB)
/// Ranges within this distance will be merged into a single request
pub const DEFAULT_GAP_THRESHOLD: u64 = 4096;

/// Batch reader that efficiently reads specific rows from a column.
///
/// Uses PageRowIndex to map row indices to byte offsets, then coalesces
/// nearby ranges and issues batch reads.
pub const BatchReader = struct {
    allocator: std.mem.Allocator,
    base_reader: Reader,
    gap_threshold: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, base_reader: Reader) Self {
        return Self{
            .allocator = allocator,
            .base_reader = base_reader,
            .gap_threshold = DEFAULT_GAP_THRESHOLD,
        };
    }

    /// Set the gap threshold for range coalescing.
    /// Larger values may improve throughput but fetch more unnecessary data.
    pub fn setGapThreshold(self: *Self, threshold: u64) void {
        self.gap_threshold = threshold;
    }

    /// Read fixed-size values at specific row indices.
    ///
    /// Parameters:
    ///   - T: The value type (i64, f64, i32, f32)
    ///   - row_index: PageRowIndex for the column
    ///   - rows: Global row indices to read (need not be sorted)
    ///
    /// Returns a slice of values in the same order as the input rows.
    /// Caller must free the returned slice.
    pub fn readFixedAtIndices(
        self: *Self,
        comptime T: type,
        row_index: *const PageRowIndex,
        rows: []const u32,
    ) ![]T {
        if (rows.len == 0) {
            return self.allocator.alloc(T, 0);
        }

        // Convert format.ByteRange to reader.ByteRange
        const format_ranges = try row_index.getByteRangesForType(T, rows, self.gap_threshold);
        defer row_index.allocator.free(format_ranges);

        // Convert to reader's ByteRange type
        const reader_ranges = try self.allocator.alloc(ByteRange, format_ranges.len);
        defer self.allocator.free(reader_ranges);

        for (format_ranges, 0..) |fr, i| {
            reader_ranges[i] = ByteRange{
                .start = fr.start,
                .end = fr.end,
            };
        }

        // Batch read all ranges
        var batch_result = try self.base_reader.batchRead(self.allocator, reader_ranges);
        defer batch_result.deinit();

        // Allocate result array
        const result = try self.allocator.alloc(T, rows.len);
        errdefer self.allocator.free(result);

        // Build a map from byte offset to buffer for efficient lookup
        // We need to map each requested row back to its data
        const value_size = @sizeOf(T);

        // For each input row, find which buffer contains its data
        for (rows, 0..) |row, result_idx| {
            const loc = row_index.getFixedRowOffset(row, value_size, 0) orelse {
                // Row out of bounds - this shouldn't happen if rows were validated
                continue;
            };

            // Find which range contains this byte offset
            var found = false;
            for (format_ranges, 0..) |range, range_idx| {
                if (loc.byte_offset >= range.start and loc.byte_offset < range.end) {
                    // Calculate offset within this buffer
                    const buf_offset: usize = @intCast(loc.byte_offset - range.start);
                    const buf = batch_result.buffers[range_idx];

                    if (buf_offset + value_size <= buf.len) {
                        // Read value from buffer
                        result[result_idx] = std.mem.readInt(T, buf[buf_offset..][0..value_size], .little);
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                // Fill with zero if not found (shouldn't happen with valid input)
                result[result_idx] = 0;
            }
        }

        return result;
    }

    /// Read f64 values at specific row indices.
    pub fn readFloat64AtIndices(
        self: *Self,
        row_index: *const PageRowIndex,
        rows: []const u32,
    ) ![]f64 {
        if (rows.len == 0) {
            return self.allocator.alloc(f64, 0);
        }

        // Get byte ranges for all rows
        const format_ranges = try row_index.getByteRangesForType(f64, rows, self.gap_threshold);
        defer row_index.allocator.free(format_ranges);

        // Convert to reader's ByteRange type
        const reader_ranges = try self.allocator.alloc(ByteRange, format_ranges.len);
        defer self.allocator.free(reader_ranges);

        for (format_ranges, 0..) |fr, i| {
            reader_ranges[i] = ByteRange{
                .start = fr.start,
                .end = fr.end,
            };
        }

        // Batch read all ranges
        var batch_result = try self.base_reader.batchRead(self.allocator, reader_ranges);
        defer batch_result.deinit();

        // Allocate result array
        const result = try self.allocator.alloc(f64, rows.len);
        errdefer self.allocator.free(result);

        const value_size = @sizeOf(f64);

        // Map each row to its value
        for (rows, 0..) |row, result_idx| {
            const loc = row_index.getFixedRowOffset(row, value_size, 0) orelse continue;

            // Find which range contains this byte offset
            for (format_ranges, 0..) |range, range_idx| {
                if (loc.byte_offset >= range.start and loc.byte_offset < range.end) {
                    const buf_offset: usize = @intCast(loc.byte_offset - range.start);
                    const buf = batch_result.buffers[range_idx];

                    if (buf_offset + value_size <= buf.len) {
                        result[result_idx] = @bitCast(std.mem.readInt(u64, buf[buf_offset..][0..8], .little));
                        break;
                    }
                }
            }
        }

        return result;
    }

    /// Read f32 values at specific row indices.
    pub fn readFloat32AtIndices(
        self: *Self,
        row_index: *const PageRowIndex,
        rows: []const u32,
    ) ![]f32 {
        if (rows.len == 0) {
            return self.allocator.alloc(f32, 0);
        }

        const format_ranges = try row_index.getByteRangesForType(f32, rows, self.gap_threshold);
        defer row_index.allocator.free(format_ranges);

        const reader_ranges = try self.allocator.alloc(ByteRange, format_ranges.len);
        defer self.allocator.free(reader_ranges);

        for (format_ranges, 0..) |fr, i| {
            reader_ranges[i] = ByteRange{
                .start = fr.start,
                .end = fr.end,
            };
        }

        var batch_result = try self.base_reader.batchRead(self.allocator, reader_ranges);
        defer batch_result.deinit();

        const result = try self.allocator.alloc(f32, rows.len);
        errdefer self.allocator.free(result);

        const value_size = @sizeOf(f32);

        for (rows, 0..) |row, result_idx| {
            const loc = row_index.getFixedRowOffset(row, value_size, 0) orelse continue;

            for (format_ranges, 0..) |range, range_idx| {
                if (loc.byte_offset >= range.start and loc.byte_offset < range.end) {
                    const buf_offset: usize = @intCast(loc.byte_offset - range.start);
                    const buf = batch_result.buffers[range_idx];

                    if (buf_offset + value_size <= buf.len) {
                        result[result_idx] = @bitCast(std.mem.readInt(u32, buf[buf_offset..][0..4], .little));
                        break;
                    }
                }
            }
        }

        return result;
    }
};

/// Statistics for batch read operations
pub const BatchReadStats = struct {
    /// Number of byte ranges requested
    num_ranges: usize,
    /// Total bytes requested
    total_bytes: u64,
    /// Number of rows retrieved
    num_rows: usize,
    /// Whether native batch read was used
    used_native_batch: bool,

    pub fn format(self: BatchReadStats) [128]u8 {
        var buf: [128]u8 = undefined;
        _ = std.fmt.bufPrint(&buf, "ranges={d}, bytes={d}, rows={d}, native={}", .{
            self.num_ranges,
            self.total_bytes,
            self.num_rows,
            self.used_native_batch,
        }) catch {};
        return buf;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "BatchReader: initialization" {
    const allocator = std.testing.allocator;

    // Create a simple memory reader for testing
    const memory_reader = @import("memory_reader.zig");
    var data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var mem_reader = memory_reader.MemoryReader.init(&data);
    const reader = mem_reader.reader();

    const batch = BatchReader.init(allocator, reader);
    _ = batch;

    // Just verify it compiles and initializes
}

test "BatchReader: gap threshold" {
    const allocator = std.testing.allocator;

    const memory_reader = @import("memory_reader.zig");
    var data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var mem_reader = memory_reader.MemoryReader.init(&data);
    const reader = mem_reader.reader();

    var batch = BatchReader.init(allocator, reader);
    try std.testing.expectEqual(@as(u64, DEFAULT_GAP_THRESHOLD), batch.gap_threshold);

    batch.setGapThreshold(8192);
    try std.testing.expectEqual(@as(u64, 8192), batch.gap_threshold);
}
