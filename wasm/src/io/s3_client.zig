//! S3/R2 Client with ETag-based Compare-And-Swap (CAS) support.
//!
//! This client supports conditional operations for distributed coordination:
//! - GET with ETag: Returns data along with its ETag
//! - PUT with If-Match: Conditional write that fails if ETag doesn't match
//!
//! This enables serverless-friendly distributed writes without external
//! coordination services (no DynamoDB/Redis locks needed).
//!
//! ## ETag-based CAS Protocol
//! ```
//! 1. Writer creates data file: data/{uuid}.lance (immediate, no lock)
//! 2. Writer GETs _versions/_latest + ETag
//! 3. Writer reads current manifest, adds new fragment
//! 4. Writer PUTs new manifest: _versions/{N+1}.manifest
//! 5. Writer PUTs _versions/_latest with If-Match: {old-etag}
//!    - Success: Done!
//!    - 412 Precondition Failed: Retry from step 2
//! ```
//!
//! NOTE: This is for native (non-WASM) builds only. For WASM/browser,
//! JavaScript handles fetch() with headers and passes results to WASM.

const std = @import("std");
const http = std.http;

/// Result of a GET request with ETag
pub const GetResult = struct {
    /// Response body data (caller owns)
    data: []u8,
    /// ETag value for conditional operations
    etag: []u8,
    /// Allocator used for data and etag
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GetResult) void {
        self.allocator.free(self.data);
        self.allocator.free(self.etag);
    }
};

/// Result of a conditional PUT operation
pub const PutResult = enum {
    /// Write succeeded
    success,
    /// 412 Precondition Failed - ETag didn't match, retry needed
    precondition_failed,
    /// Other HTTP error
    http_error,
    /// Network/connection error
    network_error,
};

/// Errors for S3 client operations
pub const S3Error = error{
    OutOfMemory,
    InvalidUrl,
    HttpError,
    NetworkError,
    NoETag,
};

/// S3/R2 client with ETag-based conditional operations.
///
/// Works with any S3-compatible storage that supports:
/// - ETag header on GET responses
/// - If-Match header on PUT requests
/// - HTTP 412 Precondition Failed response
pub const S3Client = struct {
    allocator: std.mem.Allocator,
    base_url: []const u8,

    const Self = @This();

    /// Create a new S3 client.
    ///
    /// base_url: Base URL for the dataset (e.g., "https://data.example.com/my-dataset.lance")
    pub fn init(allocator: std.mem.Allocator, base_url: []const u8) Self {
        return Self{
            .allocator = allocator,
            .base_url = base_url,
        };
    }

    /// Build full URL for a path
    fn buildUrl(self: *Self, path: []const u8) ![]const u8 {
        if (path.len > 0 and path[0] == '/') {
            return try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, path });
        } else {
            return try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.base_url, path });
        }
    }

    /// GET with ETag header returned.
    ///
    /// Returns the file content along with its ETag for use in conditional writes.
    pub fn getWithETag(self: *Self, path: []const u8) S3Error!GetResult {
        const full_url = try self.buildUrl(path);
        defer self.allocator.free(full_url);

        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = std.Uri.parse(full_url) catch return S3Error.InvalidUrl;

        var req = client.request(.GET, uri, .{}) catch return S3Error.NetworkError;
        defer req.deinit();

        req.sendBodiless() catch return S3Error.NetworkError;

        var redirect_buf: [4096]u8 = undefined;
        var response = req.receiveHead(&redirect_buf) catch return S3Error.NetworkError;

        if (response.head.status != .ok) {
            return S3Error.HttpError;
        }

        // Extract ETag from headers
        var etag: ?[]const u8 = null;
        var it = response.head.iterateHeaders();
        while (it.next()) |header| {
            if (std.ascii.eqlIgnoreCase(header.name, "etag")) {
                etag = header.value;
                break;
            }
        }

        if (etag == null) {
            return S3Error.NoETag;
        }

        // Read response body
        var body_reader = response.reader(&.{});
        var body = std.ArrayList(u8).init(self.allocator);
        errdefer body.deinit();

        // Read all data
        var buf: [8192]u8 = undefined;
        while (true) {
            const bytes = body_reader.readSliceShort(&buf) catch break;
            if (bytes == 0) break;
            try body.appendSlice(buf[0..bytes]);
        }

        // Duplicate etag since header buffer may be freed
        const etag_copy = self.allocator.dupe(u8, etag.?) catch return S3Error.OutOfMemory;
        errdefer self.allocator.free(etag_copy);

        return GetResult{
            .data = body.toOwnedSlice() catch return S3Error.OutOfMemory,
            .etag = etag_copy,
            .allocator = self.allocator,
        };
    }

    /// GET without ETag (simpler API when ETag isn't needed).
    pub fn get(self: *Self, path: []const u8) S3Error![]u8 {
        const full_url = try self.buildUrl(path);
        defer self.allocator.free(full_url);

        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = std.Uri.parse(full_url) catch return S3Error.InvalidUrl;

        var req = client.request(.GET, uri, .{}) catch return S3Error.NetworkError;
        defer req.deinit();

        req.sendBodiless() catch return S3Error.NetworkError;

        var redirect_buf: [4096]u8 = undefined;
        var response = req.receiveHead(&redirect_buf) catch return S3Error.NetworkError;

        if (response.head.status != .ok) {
            return S3Error.HttpError;
        }

        // Read response body
        var body_reader = response.reader(&.{});
        var body = std.ArrayList(u8).init(self.allocator);
        errdefer body.deinit();

        var buf: [8192]u8 = undefined;
        while (true) {
            const bytes = body_reader.readSliceShort(&buf) catch break;
            if (bytes == 0) break;
            try body.appendSlice(buf[0..bytes]);
        }

        return body.toOwnedSlice() catch return S3Error.OutOfMemory;
    }

    /// PUT with If-Match header for Compare-And-Swap.
    ///
    /// This is the key primitive for distributed coordination:
    /// - If the current ETag matches, the write succeeds
    /// - If another writer has updated the file, returns precondition_failed
    ///
    /// Returns:
    /// - success: Write completed
    /// - precondition_failed: ETag mismatch, caller should retry
    /// - http_error/network_error: Other failure
    pub fn putIfMatch(self: *Self, path: []const u8, data: []const u8, etag: []const u8) PutResult {
        const full_url = self.buildUrl(path) catch return .network_error;
        defer self.allocator.free(full_url);

        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = std.Uri.parse(full_url) catch return .network_error;

        var req = client.request(.PUT, uri, .{
            .extra_headers = &.{
                .{ .name = "If-Match", .value = etag },
                .{ .name = "Content-Type", .value = "application/octet-stream" },
            },
        }) catch return .network_error;
        defer req.deinit();

        // Send body
        req.writeAll(data) catch return .network_error;
        req.finish() catch return .network_error;

        var redirect_buf: [4096]u8 = undefined;
        const response = req.receiveHead(&redirect_buf) catch return .network_error;

        return switch (response.head.status) {
            .ok, .created, .no_content => .success,
            .precondition_failed => .precondition_failed,
            else => .http_error,
        };
    }

    /// PUT unconditional (for data files that have unique names).
    ///
    /// Use this for writing data files with UUIDs - no conflict possible.
    pub fn put(self: *Self, path: []const u8, data: []const u8) S3Error!void {
        const full_url = try self.buildUrl(path);
        defer self.allocator.free(full_url);

        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = std.Uri.parse(full_url) catch return S3Error.InvalidUrl;

        var req = client.request(.PUT, uri, .{
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/octet-stream" },
            },
        }) catch return S3Error.NetworkError;
        defer req.deinit();

        // Send body
        req.writeAll(data) catch return S3Error.NetworkError;
        req.finish() catch return S3Error.NetworkError;

        var redirect_buf: [4096]u8 = undefined;
        const response = req.receiveHead(&redirect_buf) catch return S3Error.NetworkError;

        switch (response.head.status) {
            .ok, .created, .no_content => return,
            else => return S3Error.HttpError,
        }
    }

    /// Check if a file exists.
    pub fn exists(self: *Self, path: []const u8) S3Error!bool {
        const full_url = try self.buildUrl(path);
        defer self.allocator.free(full_url);

        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = std.Uri.parse(full_url) catch return S3Error.InvalidUrl;

        var req = client.request(.HEAD, uri, .{}) catch return S3Error.NetworkError;
        defer req.deinit();

        req.sendBodiless() catch return S3Error.NetworkError;

        var redirect_buf: [4096]u8 = undefined;
        const response = req.receiveHead(&redirect_buf) catch return S3Error.NetworkError;

        return response.head.status == .ok;
    }

    /// Delete a file.
    pub fn delete(self: *Self, path: []const u8) S3Error!void {
        const full_url = try self.buildUrl(path);
        defer self.allocator.free(full_url);

        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = std.Uri.parse(full_url) catch return S3Error.InvalidUrl;

        var req = client.request(.DELETE, uri, .{}) catch return S3Error.NetworkError;
        defer req.deinit();

        req.sendBodiless() catch return S3Error.NetworkError;

        var redirect_buf: [4096]u8 = undefined;
        const response = req.receiveHead(&redirect_buf) catch return S3Error.NetworkError;

        switch (response.head.status) {
            .ok, .no_content, .not_found => return, // not_found is OK for delete
            else => return S3Error.HttpError,
        }
    }
};

/// Parse version number from _latest file content.
///
/// The _latest file contains just the version number as a string (e.g., "5").
pub fn parseVersionFromLatest(data: []const u8) ?u64 {
    // Trim whitespace
    const trimmed = std.mem.trim(u8, data, &[_]u8{ ' ', '\t', '\n', '\r' });
    if (trimmed.len == 0) return null;

    return std.fmt.parseInt(u64, trimmed, 10) catch null;
}

// ============================================================================
// Tests
// ============================================================================

test "S3Client interface compiles" {
    // Just verify the type compiles correctly
    const T = S3Client;
    _ = @TypeOf(T.init);
    _ = @TypeOf(T.getWithETag);
    _ = @TypeOf(T.putIfMatch);
}

test "parse version from latest" {
    try std.testing.expectEqual(@as(?u64, 5), parseVersionFromLatest("5"));
    try std.testing.expectEqual(@as(?u64, 123), parseVersionFromLatest("123\n"));
    try std.testing.expectEqual(@as(?u64, 42), parseVersionFromLatest("  42  \n"));
    try std.testing.expectEqual(@as(?u64, null), parseVersionFromLatest(""));
    try std.testing.expectEqual(@as(?u64, null), parseVersionFromLatest("abc"));
}

test "GetResult cleanup" {
    const allocator = std.testing.allocator;

    // Test that GetResult properly manages memory
    var result = GetResult{
        .data = try allocator.dupe(u8, "test data"),
        .etag = try allocator.dupe(u8, "\"abc123\""),
        .allocator = allocator,
    };

    // Should clean up without leaks
    result.deinit();
}
