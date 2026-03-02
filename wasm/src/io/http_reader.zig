//! HTTP Reader with Range request support.
//!
//! Reads Lance files from HTTP/HTTPS URLs using Range headers,
//! enabling reading of large files without downloading them entirely.
//!
//! NOTE: This is for native (non-WASM) builds only. For WASM/browser,
//! JavaScript handles fetch() and passes bytes to the WASM module.

const std = @import("std");
const http = std.http;
const ReaderMod = @import("reader.zig");
const Reader = ReaderMod.Reader;
const ReadError = ReaderMod.ReadError;
const ByteRange = ReaderMod.ByteRange;
const BatchReadResult = ReaderMod.BatchReadResult;

/// HTTP Reader that fetches data using Range requests.
/// Note: Creates a fresh HTTP client per request to avoid TLS connection pooling issues.
pub const HttpReader = struct {
    allocator: std.mem.Allocator,
    url: []const u8,
    file_size: u64,

    const Self = @This();

    /// Open an HTTP reader for the given URL.
    /// Performs a HEAD request to get file size.
    pub fn open(allocator: std.mem.Allocator, url: []const u8) !Self {
        var client = http.Client{ .allocator = allocator };
        defer client.deinit();

        // Parse URL
        const uri = std.Uri.parse(url) catch return error.InvalidUrl;

        // HEAD request to get file size
        var req = try client.request(.HEAD, uri, .{});
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buf: [4096]u8 = undefined;
        const response = try req.receiveHead(&redirect_buf);

        if (response.head.status != .ok) {
            return error.HttpError;
        }

        // Get content length
        const content_length = response.head.content_length orelse return error.NoContentLength;

        return Self{
            .allocator = allocator,
            .url = url,
            .file_size = content_length,
        };
    }

    pub fn deinit(self: *Self) void {
        // Nothing to clean up - each request uses its own client
        _ = self;
    }

    /// Read bytes at offset using HTTP Range request.
    pub fn readAt(self: *Self, offset: u64, buffer: []u8) !usize {
        // Create fresh client per request to avoid TLS connection pooling issues
        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = std.Uri.parse(self.url) catch return error.InvalidUrl;

        // Create Range header value: "bytes=offset-end"
        var range_buf: [64]u8 = undefined;
        const end = offset + buffer.len - 1;
        const range_str = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ offset, end }) catch return error.InvalidRange;

        var req = try client.request(.GET, uri, .{
            .extra_headers = &.{
                .{ .name = "Range", .value = range_str },
            },
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buf: [4096]u8 = undefined;
        var response = try req.receiveHead(&redirect_buf);

        // Accept both 200 (full content) and 206 (partial content)
        if (response.head.status != .ok and response.head.status != .partial_content) {
            return error.HttpError;
        }

        // Read response body
        var body_reader = response.reader(&.{});
        var total_read: usize = 0;
        while (total_read < buffer.len) {
            const bytes = body_reader.readSliceShort(buffer[total_read..]) catch break;
            if (bytes == 0) break;
            total_read += bytes;
        }
        return total_read;
    }

    /// Get file size.
    pub fn getSize(self: *Self) u64 {
        return self.file_size;
    }

    /// Read multiple byte ranges.
    ///
    /// For maximum compatibility, this makes sequential requests for each range.
    /// Note: HTTP/1.1 supports multipart/byteranges but not all servers implement it.
    /// Future optimization: detect server support and use true multi-range requests.
    pub fn batchRead(self: *Self, allocator: std.mem.Allocator, ranges: []const ByteRange) !BatchReadResult {
        if (ranges.len == 0) {
            return BatchReadResult{
                .buffers = try allocator.alloc([]u8, 0),
                .allocator = allocator,
            };
        }

        // Allocate result buffers
        const buffers = try allocator.alloc([]u8, ranges.len);
        errdefer allocator.free(buffers);

        var completed: usize = 0;
        errdefer {
            for (buffers[0..completed]) |buf| {
                allocator.free(buf);
            }
        }

        // Read each range sequentially
        // Note: Could be parallelized with threads, but keeping simple for now
        for (ranges, 0..) |range, i| {
            const len: usize = @intCast(range.len());
            const buffer = try allocator.alloc(u8, len);
            errdefer allocator.free(buffer);

            // Read this range
            const bytes_read = try self.readAt(range.start, buffer);
            if (bytes_read < len) {
                // Partial read - resize buffer
                const actual = try allocator.realloc(buffer, bytes_read);
                buffers[i] = actual;
            } else {
                buffers[i] = buffer;
            }
            completed += 1;
        }

        return BatchReadResult{
            .buffers = buffers,
            .allocator = allocator,
        };
    }

    // ========================================================================
    // Reader VTable implementation
    // ========================================================================

    fn vtableRead(ptr: *anyopaque, offset: u64, buffer: []u8) ReadError!usize {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.readAt(offset, buffer) catch return ReadError.NetworkError;
    }

    fn vtableSize(ptr: *anyopaque) ReadError!u64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.file_size;
    }

    fn vtableDeinit(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    fn vtableBatchRead(ptr: *anyopaque, allocator: std.mem.Allocator, ranges: []const ByteRange) ReadError!BatchReadResult {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.batchRead(allocator, ranges) catch return ReadError.NetworkError;
    }

    const vtable = Reader.VTable{
        .read = vtableRead,
        .size = vtableSize,
        .deinit = vtableDeinit,
        .batch_read = vtableBatchRead,
    };

    /// Create a Reader interface from this HttpReader.
    pub fn reader(self: *Self) Reader {
        return Reader{
            .ptr = self,
            .vtable = &vtable,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "HttpReader interface compiles" {
    // Just verify the type compiles correctly
    const T = HttpReader;
    _ = T.vtable;
}
