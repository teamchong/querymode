//! HTTP/1.1 Request/Response Parser
//!
//! Minimal HTTP implementation for the serve command.
//! Supports GET and POST methods with JSON bodies.

const std = @import("std");

/// HTTP request method
pub const Method = enum {
    GET,
    POST,
    OPTIONS,
    HEAD,
    PUT,
    DELETE,
    UNKNOWN,

    pub fn fromString(s: []const u8) Method {
        if (std.mem.eql(u8, s, "GET")) return .GET;
        if (std.mem.eql(u8, s, "POST")) return .POST;
        if (std.mem.eql(u8, s, "OPTIONS")) return .OPTIONS;
        if (std.mem.eql(u8, s, "HEAD")) return .HEAD;
        if (std.mem.eql(u8, s, "PUT")) return .PUT;
        if (std.mem.eql(u8, s, "DELETE")) return .DELETE;
        return .UNKNOWN;
    }
};

/// Parsed HTTP request
pub const Request = struct {
    method: Method,
    path: []const u8,
    query: ?[]const u8,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
    raw: []const u8,

    pub fn deinit(self: *Request) void {
        self.headers.deinit();
    }

    /// Get a header value (case-insensitive key lookup)
    pub fn getHeader(self: *const Request, key: []const u8) ?[]const u8 {
        // Try exact match first
        if (self.headers.get(key)) |v| return v;

        // Try lowercase
        var lower_buf: [64]u8 = undefined;
        if (key.len <= lower_buf.len) {
            for (key, 0..) |c, i| {
                lower_buf[i] = std.ascii.toLower(c);
            }
            return self.headers.get(lower_buf[0..key.len]);
        }
        return null;
    }

    /// Get Content-Length header as integer
    pub fn getContentLength(self: *const Request) ?usize {
        const value = self.getHeader("Content-Length") orelse
            self.getHeader("content-length") orelse return null;
        return std.fmt.parseInt(usize, value, 10) catch null;
    }
};

/// HTTP response builder
pub const Response = struct {
    status: u16,
    status_text: []const u8,
    headers: std.ArrayListUnmanaged(Header),
    body: []const u8,
    allocator: std.mem.Allocator,

    const Header = struct {
        name: []const u8,
        value: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator) Response {
        return .{
            .status = 200,
            .status_text = "OK",
            .headers = std.ArrayListUnmanaged(Header){},
            .body = "",
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Response) void {
        self.headers.deinit(self.allocator);
    }

    pub fn setStatus(self: *Response, status: u16, text: []const u8) void {
        self.status = status;
        self.status_text = text;
    }

    pub fn addHeader(self: *Response, name: []const u8, value: []const u8) !void {
        try self.headers.append(self.allocator, .{ .name = name, .value = value });
    }

    pub fn setBody(self: *Response, body: []const u8) void {
        self.body = body;
    }

    /// Serialize response to bytes
    pub fn toBytes(self: *Response) ![]u8 {
        var buffer = std.ArrayListUnmanaged(u8){};
        errdefer buffer.deinit(self.allocator);

        // Status line
        try buffer.writer(self.allocator).print("HTTP/1.1 {} {s}\r\n", .{ self.status, self.status_text });

        // Add Content-Length if body present
        if (self.body.len > 0) {
            try buffer.writer(self.allocator).print("Content-Length: {}\r\n", .{self.body.len});
        }

        // Headers
        for (self.headers.items) |header| {
            try buffer.writer(self.allocator).print("{s}: {s}\r\n", .{ header.name, header.value });
        }

        // Empty line + body
        try buffer.appendSlice(self.allocator, "\r\n");
        try buffer.appendSlice(self.allocator, self.body);

        return buffer.toOwnedSlice(self.allocator);
    }
};

/// Parse HTTP request from raw bytes
pub fn parseRequest(allocator: std.mem.Allocator, data: []const u8) !Request {
    var headers = std.StringHashMap([]const u8).init(allocator);
    errdefer headers.deinit();

    // Find end of headers
    const header_end = std.mem.indexOf(u8, data, "\r\n\r\n") orelse
        return error.InvalidRequest;

    const header_section = data[0..header_end];
    const body = if (header_end + 4 < data.len) data[header_end + 4 ..] else "";

    // Parse request line
    var lines = std.mem.splitSequence(u8, header_section, "\r\n");
    const request_line = lines.next() orelse return error.InvalidRequest;

    var parts = std.mem.splitScalar(u8, request_line, ' ');
    const method_str = parts.next() orelse return error.InvalidRequest;
    const full_path = parts.next() orelse return error.InvalidRequest;

    // Parse path and query string
    var path: []const u8 = full_path;
    var query: ?[]const u8 = null;

    if (std.mem.indexOf(u8, full_path, "?")) |q_idx| {
        path = full_path[0..q_idx];
        query = full_path[q_idx + 1 ..];
    }

    // Parse headers
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (std.mem.indexOf(u8, line, ": ")) |colon_idx| {
            const name = line[0..colon_idx];
            const value = line[colon_idx + 2 ..];
            try headers.put(name, value);
        }
    }

    return Request{
        .method = Method.fromString(method_str),
        .path = path,
        .query = query,
        .headers = headers,
        .body = body,
        .raw = data,
    };
}

/// Common HTTP responses
pub fn jsonResponse(allocator: std.mem.Allocator, json: []const u8) !Response {
    var resp = Response.init(allocator);
    try resp.addHeader("Content-Type", "application/json");
    try resp.addHeader("Access-Control-Allow-Origin", "*");
    resp.setBody(json);
    return resp;
}

pub fn htmlResponse(allocator: std.mem.Allocator, html: []const u8) !Response {
    var resp = Response.init(allocator);
    try resp.addHeader("Content-Type", "text/html; charset=utf-8");
    resp.setBody(html);
    return resp;
}

pub fn errorResponse(allocator: std.mem.Allocator, status: u16, message: []const u8) !Response {
    var resp = Response.init(allocator);
    resp.setStatus(status, switch (status) {
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        else => "Error",
    });
    try resp.addHeader("Content-Type", "application/json");
    try resp.addHeader("Access-Control-Allow-Origin", "*");

    const json = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\"}}", .{message});
    resp.setBody(json);
    return resp;
}

pub fn corsPreflightResponse(allocator: std.mem.Allocator) !Response {
    var resp = Response.init(allocator);
    resp.setStatus(204, "No Content");
    try resp.addHeader("Access-Control-Allow-Origin", "*");
    try resp.addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    try resp.addHeader("Access-Control-Allow-Headers", "Content-Type");
    try resp.addHeader("Access-Control-Max-Age", "86400");
    return resp;
}

// =============================================================================
// Tests
// =============================================================================

test "parse simple GET request" {
    const allocator = std.testing.allocator;

    const raw =
        "GET /api/query HTTP/1.1\r\n" ++
        "Host: localhost:3000\r\n" ++
        "Content-Type: application/json\r\n" ++
        "\r\n";

    var req = try parseRequest(allocator, raw);
    defer req.deinit();

    try std.testing.expectEqual(Method.GET, req.method);
    try std.testing.expectEqualStrings("/api/query", req.path);
    try std.testing.expectEqualStrings("localhost:3000", req.headers.get("Host").?);
}

test "parse POST request with body" {
    const allocator = std.testing.allocator;

    const raw =
        "POST /api/query HTTP/1.1\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: 25\r\n" ++
        "\r\n" ++
        "{\"sql\":\"SELECT * LIMIT 1\"}";

    var req = try parseRequest(allocator, raw);
    defer req.deinit();

    try std.testing.expectEqual(Method.POST, req.method);
    try std.testing.expectEqualStrings("/api/query", req.path);
    try std.testing.expectEqualStrings("{\"sql\":\"SELECT * LIMIT 1\"}", req.body);
}

test "parse request with query string" {
    const allocator = std.testing.allocator;

    const raw =
        "GET /api/data?limit=10&offset=20 HTTP/1.1\r\n" ++
        "\r\n";

    var req = try parseRequest(allocator, raw);
    defer req.deinit();

    try std.testing.expectEqualStrings("/api/data", req.path);
    try std.testing.expectEqualStrings("limit=10&offset=20", req.query.?);
}

test "build response" {
    const allocator = std.testing.allocator;

    var resp = try jsonResponse(allocator, "{\"ok\":true}");
    defer resp.deinit();

    const bytes = try resp.toBytes();
    defer allocator.free(bytes);

    try std.testing.expect(std.mem.indexOf(u8, bytes, "HTTP/1.1 200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "Content-Type: application/json") != null);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "{\"ok\":true}") != null);
}
