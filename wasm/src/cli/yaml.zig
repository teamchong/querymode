//! Simple YAML Parser for LanceQL Config Files
//!
//! Parses config format:
//!   command: ingest
//!   input: "data.csv"
//!   output: "out.lance"
//!   options:
//!     format: csv
//!     header: true

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    allocator: Allocator,
    command: []const u8,
    input: ?[]const u8,
    output: ?[]const u8,
    options: std.StringHashMap([]const u8),

    const Self = @This();

    pub fn deinit(self: *Self) void {
        if (self.command.len > 0) self.allocator.free(self.command);
        if (self.input) |v| self.allocator.free(v);
        if (self.output) |v| self.allocator.free(v);

        var it = self.options.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.options.deinit();
    }
};

pub const ParseError = error{
    InvalidFormat,
    MissingCommand,
    OutOfMemory,
};

pub fn parse(allocator: Allocator, content: []const u8) ParseError!Config {
    var config = Config{
        .allocator = allocator,
        .command = "",
        .input = null,
        .output = null,
        .options = std.StringHashMap([]const u8).init(allocator),
    };
    errdefer config.deinit();

    var in_options = false;
    var lines = std.mem.splitScalar(u8, content, '\n');

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;

        // Check indentation for options block
        const indent = getIndent(line);
        if (indent > 0 and in_options) {
            // This is an option
            if (parseKeyValue(trimmed)) |kv| {
                const key = allocator.dupe(u8, kv.key) catch return ParseError.OutOfMemory;
                errdefer allocator.free(key);
                const value = allocator.dupe(u8, kv.value) catch return ParseError.OutOfMemory;
                config.options.put(key, value) catch return ParseError.OutOfMemory;
            }
            continue;
        }

        in_options = false;

        if (parseKeyValue(trimmed)) |kv| {
            if (std.mem.eql(u8, kv.key, "command")) {
                config.command = allocator.dupe(u8, kv.value) catch return ParseError.OutOfMemory;
            } else if (std.mem.eql(u8, kv.key, "input")) {
                config.input = allocator.dupe(u8, kv.value) catch return ParseError.OutOfMemory;
            } else if (std.mem.eql(u8, kv.key, "output")) {
                config.output = allocator.dupe(u8, kv.value) catch return ParseError.OutOfMemory;
            } else if (std.mem.eql(u8, kv.key, "options")) {
                in_options = true;
            } else if (std.mem.eql(u8, kv.key, "embed")) {
                const key = allocator.dupe(u8, "embed") catch return ParseError.OutOfMemory;
                errdefer allocator.free(key);
                const value = allocator.dupe(u8, kv.value) catch return ParseError.OutOfMemory;
                config.options.put(key, value) catch return ParseError.OutOfMemory;
            } else if (std.mem.eql(u8, kv.key, "model")) {
                const key = allocator.dupe(u8, "model") catch return ParseError.OutOfMemory;
                errdefer allocator.free(key);
                const value = allocator.dupe(u8, kv.value) catch return ParseError.OutOfMemory;
                config.options.put(key, value) catch return ParseError.OutOfMemory;
            }
        }
    }

    if (config.command.len == 0) {
        return ParseError.MissingCommand;
    }

    return config;
}

fn getIndent(line: []const u8) usize {
    var indent: usize = 0;
    for (line) |c| {
        if (c == ' ') {
            indent += 1;
        } else if (c == '\t') {
            indent += 2;
        } else {
            break;
        }
    }
    return indent;
}

const KeyValue = struct { key: []const u8, value: []const u8 };

fn parseKeyValue(line: []const u8) ?KeyValue {
    const colon_pos = std.mem.indexOf(u8, line, ":") orelse return null;
    if (colon_pos == 0) return null;

    const key = std.mem.trim(u8, line[0..colon_pos], " \t");
    var value = std.mem.trim(u8, line[colon_pos + 1 ..], " \t");

    // Strip quotes
    if (value.len >= 2) {
        if ((value[0] == '"' and value[value.len - 1] == '"') or
            (value[0] == '\'' and value[value.len - 1] == '\''))
        {
            value = value[1 .. value.len - 1];
        }
    }

    return .{ .key = key, .value = value };
}

// =============================================================================
// Tests
// =============================================================================

test "parse simple config" {
    const allocator = std.testing.allocator;
    const content =
        \\command: ingest
        \\input: "data.csv"
        \\output: "out.lance"
    ;

    var config = try parse(allocator, content);
    defer config.deinit();

    try std.testing.expectEqualStrings("ingest", config.command);
    try std.testing.expectEqualStrings("data.csv", config.input.?);
    try std.testing.expectEqualStrings("out.lance", config.output.?);
}

test "parse config with options" {
    const allocator = std.testing.allocator;
    const content =
        \\command: ingest
        \\input: data.csv
        \\output: out.lance
        \\options:
        \\  format: csv
        \\  header: true
    ;

    var config = try parse(allocator, content);
    defer config.deinit();

    try std.testing.expectEqualStrings("ingest", config.command);
    try std.testing.expectEqualStrings("csv", config.options.get("format").?);
    try std.testing.expectEqualStrings("true", config.options.get("header").?);
}

test "parse enrich config" {
    const allocator = std.testing.allocator;
    const content =
        \\command: enrich
        \\input: data.lance
        \\embed: "text"
        \\model: minilm
    ;

    var config = try parse(allocator, content);
    defer config.deinit();

    try std.testing.expectEqualStrings("enrich", config.command);
    try std.testing.expectEqualStrings("text", config.options.get("embed").?);
    try std.testing.expectEqualStrings("minilm", config.options.get("model").?);
}

test "missing command returns error" {
    const allocator = std.testing.allocator;
    const content =
        \\input: data.csv
        \\output: out.lance
    ;

    const result = parse(allocator, content);
    try std.testing.expectError(ParseError.MissingCommand, result);
}
