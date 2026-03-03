//! WASM bindings for Lance dataset writer with CAS coordination.
//!
//! Provides JavaScript-callable exports for Cloudflare Workers and browser environments.
//! JavaScript handles HTTP fetch with ETag headers and passes results to WASM.
//!
//! ## JavaScript Usage (Cloudflare Worker)
//! ```javascript
//! import { DatasetWriter } from './edgeq.wasm';
//!
//! export default {
//!   async fetch(request) {
//!     const writer = new DatasetWriter('https://r2.example.com/test.lance');
//!     await writer.append({ id: [1, 2, 3], value: [1.5, 2.5, 3.5] });
//!     return new Response('OK');
//!   }
//! };
//! ```
//!
//! ## Protocol
//! 1. JS calls datasetWriterInit(url) to initialize
//! 2. JS calls fragmentBegin() to start a new data file
//! 3. JS calls fragmentAdd*Column() to add data
//! 4. JS calls fragmentEnd() to finalize the data
//! 5. JS performs the CAS loop using jsFetch imports:
//!    - PUT data file
//!    - GET _latest with ETag
//!    - PUT new manifest
//!    - PUT _latest with If-Match

const std = @import("std");
const lance_writer = @import("lance_writer.zig");
const memory = @import("memory.zig");

// Import JavaScript fetch functions
extern fn js_log(ptr: [*]const u8, len: usize) void;

// JavaScript provides these functions for HTTP operations
// These are called by the JS side, not directly by WASM
// WASM just prepares the data and JS handles the actual fetch

// ============================================================================
// State Management
// ============================================================================

const MAX_URL_LEN = 2048;
const MAX_PATH_LEN = 256;

var dataset_url_buf: [MAX_URL_LEN]u8 = undefined;
var dataset_url_len: usize = 0;

var current_data_path_buf: [MAX_PATH_LEN]u8 = undefined;
var current_data_path_len: usize = 0;

var current_manifest_buf: [64 * 1024]u8 = undefined;
var current_manifest_len: usize = 0;

var cas_retry_count: u32 = 0;
const MAX_CAS_RETRIES: u32 = 10;

// ============================================================================
// Dataset Writer API
// ============================================================================

/// Initialize dataset writer with base URL.
/// Returns 1 on success, 0 on failure.
pub export fn datasetWriterInit(url_ptr: [*]const u8, url_len: usize) u32 {
    js_log("datasetWriterInit: enter", 24);

    if (url_len > MAX_URL_LEN) {
        js_log("datasetWriterInit: url too long", 30);
        return 0;
    }

    // Copy URL to buffer
    var i: usize = 0;
    while (i < url_len) : (i += 1) {
        dataset_url_buf[i] = url_ptr[i];
    }
    dataset_url_len = url_len;

    cas_retry_count = 0;

    js_log("datasetWriterInit: done", 23);
    return 1;
}

/// Get the base dataset URL pointer (for JS to build paths).
pub export fn datasetWriterGetUrl() [*]const u8 {
    return &dataset_url_buf;
}

/// Get the base dataset URL length.
pub export fn datasetWriterGetUrlLen() usize {
    return dataset_url_len;
}

// Simple counter for UUID generation in WASM (not truly random)
// In production, JS should generate proper UUIDs and pass them in
var uuid_counter: u64 = 0;

/// Set UUID counter seed (should be called from JS with Date.now() or similar).
pub export fn setUUIDSeed(seed: u64) void {
    uuid_counter = seed;
}

/// Generate a UUID for data file naming.
/// Writes UUID to provided buffer, returns length.
/// NOTE: For production, JS should generate real UUIDs and pass them via setCurrentDataPath.
pub export fn generateDataFileUUID(out_ptr: [*]u8, max_len: usize) usize {
    if (max_len < 36) return 0;

    // Increment counter for uniqueness
    uuid_counter +%= 1;
    const val = uuid_counter;

    const chars = "0123456789abcdef";

    // Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    // Simple generation based on counter
    const shifts = [_]u6{ 0, 4, 8, 12, 16, 20, 24, 28 };

    // First 8 hex chars
    for (0..8) |j| {
        const shift_idx = j % shifts.len;
        out_ptr[j] = chars[@as(usize, @truncate((val >> shifts[shift_idx]) & 0xF))];
    }
    out_ptr[8] = '-';

    // Next 4 hex chars
    for (0..4) |j| {
        const shift_idx = (j + 2) % shifts.len;
        out_ptr[9 + j] = chars[@as(usize, @truncate((val >> shifts[shift_idx]) & 0xF))];
    }
    out_ptr[13] = '-';

    // 4xxx (version 4)
    out_ptr[14] = '4';
    for (0..3) |j| {
        const shift_idx = (j + 4) % shifts.len;
        out_ptr[15 + j] = chars[@as(usize, @truncate((val >> shifts[shift_idx]) & 0xF))];
    }
    out_ptr[18] = '-';

    // yxxx (variant)
    out_ptr[19] = '8';
    for (0..3) |j| {
        const shift_idx = (j + 6) % shifts.len;
        out_ptr[20 + j] = chars[@as(usize, @truncate((val >> shifts[shift_idx]) & 0xF))];
    }
    out_ptr[23] = '-';

    // Last 12 hex chars
    for (0..12) |j| {
        const shift_idx = j % shifts.len;
        out_ptr[24 + j] = chars[@as(usize, @truncate((val >> shifts[shift_idx]) & 0xF))];
    }

    return 36;
}

/// Set the current data file path (called by JS after generating UUID).
pub export fn setCurrentDataPath(path_ptr: [*]const u8, path_len: usize) u32 {
    if (path_len > MAX_PATH_LEN) return 0;

    var i: usize = 0;
    while (i < path_len) : (i += 1) {
        current_data_path_buf[i] = path_ptr[i];
    }
    current_data_path_len = path_len;

    return 1;
}

/// Get the current data file path pointer.
pub export fn getCurrentDataPath() [*]const u8 {
    return &current_data_path_buf;
}

/// Get the current data file path length.
pub export fn getCurrentDataPathLen() usize {
    return current_data_path_len;
}

// ============================================================================
// Manifest Building
// ============================================================================

/// Start building a new manifest.
/// version: New version number
pub export fn manifestBegin(version: u64) u32 {
    js_log("manifestBegin: enter", 20);

    current_manifest_len = 0;

    // Write protobuf header placeholder (will fill in length later)
    // For now, we'll use a simplified manifest format

    _ = version;

    js_log("manifestBegin: done", 19);
    return 1;
}

/// Add a fragment to the manifest.
/// Returns 1 on success, 0 on failure.
pub export fn manifestAddFragment(
    fragment_id: u64,
    path_ptr: [*]const u8,
    path_len: usize,
    physical_rows: u64,
) u32 {
    js_log("manifestAddFragment: enter", 26);

    _ = fragment_id;
    _ = path_ptr;
    _ = path_len;
    _ = physical_rows;

    // In a full implementation, this would build the protobuf manifest
    // For now, we'll rely on JS to handle manifest building

    js_log("manifestAddFragment: done", 25);
    return 1;
}

/// Finalize manifest and get the bytes.
/// Returns the length of the manifest.
pub export fn manifestFinalize() usize {
    js_log("manifestFinalize: enter", 23);

    // In a full implementation, this would complete the protobuf encoding
    // For now, return placeholder

    js_log("manifestFinalize: done", 22);
    return current_manifest_len;
}

/// Get pointer to manifest buffer.
pub export fn manifestGetBuffer() [*]u8 {
    return &current_manifest_buf;
}

// ============================================================================
// CAS Retry Management
// ============================================================================

/// Increment CAS retry counter.
/// Returns 1 if more retries allowed, 0 if max exceeded.
pub export fn casRetry() u32 {
    cas_retry_count += 1;
    if (cas_retry_count >= MAX_CAS_RETRIES) {
        return 0; // Max retries exceeded
    }
    return 1;
}

/// Reset CAS retry counter (call on success).
pub export fn casReset() void {
    cas_retry_count = 0;
}

/// Get current retry count.
pub export fn casGetRetryCount() u32 {
    return cas_retry_count;
}

// ============================================================================
// Helper: Parse version from _latest content
// ============================================================================

/// Parse version number from _latest file content.
/// Returns version number or 0 on failure.
pub export fn parseLatestVersion(data_ptr: [*]const u8, data_len: usize) u64 {
    if (data_len == 0) return 0;

    // Find first non-whitespace
    var start: usize = 0;
    while (start < data_len and (data_ptr[start] == ' ' or data_ptr[start] == '\t' or data_ptr[start] == '\n' or data_ptr[start] == '\r')) {
        start += 1;
    }

    if (start >= data_len) return 0;

    // Find end of number
    var end = start;
    while (end < data_len and data_ptr[end] >= '0' and data_ptr[end] <= '9') {
        end += 1;
    }

    if (end == start) return 0;

    // Parse number
    var result: u64 = 0;
    var i = start;
    while (i < end) : (i += 1) {
        result = result * 10 + (data_ptr[i] - '0');
    }

    return result;
}

/// Format version number to string.
/// Returns length written.
pub export fn formatVersion(version: u64, out_ptr: [*]u8, max_len: usize) usize {
    if (max_len == 0) return 0;

    if (version == 0) {
        out_ptr[0] = '0';
        return 1;
    }

    // Count digits
    var temp = version;
    var digit_count: usize = 0;
    while (temp > 0) : (temp /= 10) {
        digit_count += 1;
    }

    if (digit_count > max_len) return 0;

    // Write digits in reverse
    temp = version;
    var i = digit_count;
    while (i > 0) : (i -= 1) {
        out_ptr[i - 1] = @as(u8, @truncate(temp % 10)) + '0';
        temp /= 10;
    }

    return digit_count;
}

// ============================================================================
// Re-export fragment writer functions
// ============================================================================
// The fragment writer (lance_writer.zig) handles the actual data file creation.
// These are re-exported for convenience.

pub const fragmentBegin = lance_writer.fragmentBegin;
pub const fragmentAddInt64Column = lance_writer.fragmentAddInt64Column;
pub const fragmentAddInt32Column = lance_writer.fragmentAddInt32Column;
pub const fragmentAddFloat64Column = lance_writer.fragmentAddFloat64Column;
pub const fragmentAddFloat32Column = lance_writer.fragmentAddFloat32Column;
pub const fragmentAddStringColumn = lance_writer.fragmentAddStringColumn;
pub const fragmentAddVectorColumn = lance_writer.fragmentAddVectorColumn;
pub const fragmentAddBoolColumn = lance_writer.fragmentAddBoolColumn;
pub const fragmentEnd = lance_writer.fragmentEnd;
pub const writerGetBuffer = lance_writer.writerGetBuffer;
pub const writerGetOffset = lance_writer.writerGetOffset;

// ============================================================================
// Tests (for native builds)
// ============================================================================

test "parse latest version" {
    try std.testing.expectEqual(@as(u64, 5), parseLatestVersion("5", 1));
    try std.testing.expectEqual(@as(u64, 123), parseLatestVersion("123\n", 4));
    try std.testing.expectEqual(@as(u64, 42), parseLatestVersion("  42  ", 6));
    try std.testing.expectEqual(@as(u64, 0), parseLatestVersion("", 0));
}

test "format version" {
    var buf: [32]u8 = undefined;

    const len1 = formatVersion(0, &buf, 32);
    try std.testing.expectEqual(@as(usize, 1), len1);
    try std.testing.expectEqualSlices(u8, "0", buf[0..1]);

    const len2 = formatVersion(123, &buf, 32);
    try std.testing.expectEqual(@as(usize, 3), len2);
    try std.testing.expectEqualSlices(u8, "123", buf[0..3]);

    const len3 = formatVersion(9876543210, &buf, 32);
    try std.testing.expectEqual(@as(usize, 10), len3);
    try std.testing.expectEqualSlices(u8, "9876543210", buf[0..10]);
}
