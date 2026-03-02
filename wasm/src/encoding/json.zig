//! JSON/JSONL Parser for LanceQL
//!
//! Parses JSON arrays and JSONL (newline-delimited JSON) into columnar data.
//!
//! Features:
//! - Auto-detect format (JSON array vs JSONL)
//! - Type inference (int64, float64, bool, string)
//! - Nested object flattening with dot notation
//! - Type promotion (int→float, mixed→string)
//!
//! Usage:
//!   const result = try json.readJson(allocator, data, .{});
//!   defer {
//!       for (result.columns) |*col| col.deinit();
//!       allocator.free(result.columns);
//!   }

const std = @import("std");
const csv = @import("csv.zig");

pub const JsonError = error{
    InvalidJson,
    InconsistentSchema,
    NestingTooDeep,
    OutOfMemory,
    EmptyInput,
    UnexpectedToken,
};

/// Detected JSON format
pub const Format = enum {
    json_array, // [{...}, {...}]
    jsonl, // {...}\n{...}
    empty,
    unknown,

    pub fn format(self: Format) []const u8 {
        return switch (self) {
            .json_array => "json_array",
            .jsonl => "jsonl",
            .empty => "empty",
            .unknown => "unknown",
        };
    }
};

/// Inferred column type (mirrors csv.ColumnType for compatibility)
pub const ColumnType = enum {
    int64,
    float64,
    bool_,
    string,
    null_, // All values null (promotes to string)

    pub fn format(self: ColumnType) []const u8 {
        return switch (self) {
            .int64 => "int64",
            .float64 => "float64",
            .bool_ => "bool",
            .string => "string",
            .null_ => "null",
        };
    }

    /// Promote types when merging: int+float→float, any+null→any, mixed→string
    pub fn promote(self: ColumnType, other: ColumnType) ColumnType {
        if (self == other) return self;
        if (self == .null_) return other;
        if (other == .null_) return self;

        // int64 + float64 → float64
        if ((self == .int64 and other == .float64) or
            (self == .float64 and other == .int64))
        {
            return .float64;
        }

        // Incompatible types → string
        return .string;
    }

    /// Convert to csv.ColumnType for ColumnData compatibility
    pub fn toCsvType(self: ColumnType) csv.ColumnType {
        return switch (self) {
            .int64 => .int64,
            .float64 => .float64,
            .bool_ => .bool_,
            .string, .null_ => .string,
        };
    }
};

/// Parser configuration
pub const Config = struct {
    max_nesting_depth: u32 = 10,
    sample_rows: usize = 100,
};

/// Column info discovered from JSON
pub const ColumnInfo = struct {
    name: []const u8,
    col_type: ColumnType,
    nullable: bool,
};

/// Re-use csv.ColumnData for output compatibility
pub const ColumnData = csv.ColumnData;

/// Result from JSON parsing
pub const JsonResult = struct {
    columns: []ColumnData,
    row_count: usize,
};

/// Detect JSON format from data
pub fn detectFormat(data: []const u8) Format {
    const trimmed = std.mem.trimLeft(u8, data, " \t\n\r");
    if (trimmed.len == 0) return .empty;

    // JSON array starts with '['
    if (trimmed[0] == '[') return .json_array;

    // JSONL starts with '{' (object per line)
    if (trimmed[0] == '{') return .jsonl;

    return .unknown;
}

/// Parsed JSON value
const JsonValue = union(enum) {
    null_,
    bool_: bool,
    int_: i64,
    float_: f64,
    string_: []const u8,
};

/// Key-value pair from flattened object
const KeyValue = struct {
    key: []const u8,
    value: JsonValue,
};

/// Get type from a JSON value
fn valueToType(value: std.json.Value) ColumnType {
    return switch (value) {
        .null => .null_,
        .bool => .bool_,
        .integer => .int64,
        .float => .float64,
        .string => .string,
        .array, .object => .string, // Serialize nested structures as strings
        .number_string => .float64,
    };
}

/// Convert JSON value to our JsonValue union
/// Note: Strings are NOT duplicated here - caller must handle memory
fn convertValue(value: std.json.Value) JsonValue {
    return switch (value) {
        .null => .null_,
        .bool => |b| .{ .bool_ = b },
        .integer => |i| .{ .int_ = i },
        .float => |f| .{ .float_ = f },
        .string => |s| .{ .string_ = s },
        .number_string => |s| blk: {
            // Try parsing as float
            const f = std.fmt.parseFloat(f64, s) catch 0.0;
            break :blk .{ .float_ = f };
        },
        .array, .object => .{ .string_ = "[complex]" }, // Placeholder
    };
}

/// Convert JSON value and duplicate strings
fn convertValueOwned(allocator: std.mem.Allocator, value: std.json.Value) !JsonValue {
    return switch (value) {
        .null => .null_,
        .bool => |b| .{ .bool_ = b },
        .integer => |i| .{ .int_ = i },
        .float => |f| .{ .float_ = f },
        .string => |s| .{ .string_ = try allocator.dupe(u8, s) },
        .number_string => |s| blk: {
            const f = std.fmt.parseFloat(f64, s) catch 0.0;
            break :blk .{ .float_ = f };
        },
        .array, .object => .{ .string_ = "[complex]" },
    };
}

/// Flatten a JSON object into key-value pairs with dot notation
/// Keys and string values are duplicated to ensure they outlive the JSON parse tree
fn flattenObject(
    allocator: std.mem.Allocator,
    obj: std.json.Value,
    prefix: []const u8,
    result: *std.ArrayList(KeyValue),
    depth: u32,
    max_depth: u32,
) !void {
    if (depth > max_depth) return JsonError.NestingTooDeep;

    switch (obj) {
        .object => |map| {
            var iter = map.iterator();
            while (iter.next()) |entry| {
                const key = entry.key_ptr.*;
                const full_key = if (prefix.len == 0)
                    try allocator.dupe(u8, key) // Always duplicate keys
                else
                    try std.fmt.allocPrint(allocator, "{s}.{s}", .{ prefix, key });

                const val = entry.value_ptr.*;
                switch (val) {
                    .object => {
                        try flattenObject(allocator, val, full_key, result, depth + 1, max_depth);
                        // Free intermediate key since it was only used for recursion
                        allocator.free(full_key);
                    },
                    else => {
                        try result.append(allocator, .{
                            .key = full_key,
                            .value = try convertValueOwned(allocator, val),
                        });
                    },
                }
            }
        },
        else => {
            // Non-object at top level
            const key = if (prefix.len == 0) try allocator.dupe(u8, "value") else try allocator.dupe(u8, prefix);
            try result.append(allocator, .{
                .key = key,
                .value = try convertValueOwned(allocator, obj),
            });
        },
    }
}

/// Parse JSONL data (one object per line)
fn parseJsonl(
    allocator: std.mem.Allocator,
    data: []const u8,
    config: Config,
) !JsonResult {
    var column_map = std.StringHashMap(ColumnInfo).init(allocator);
    defer column_map.deinit();

    var column_order = std.ArrayList([]const u8){};
    defer column_order.deinit(allocator);

    var all_rows = std.ArrayList([]KeyValue){};
    defer {
        for (all_rows.items) |row| {
            for (row) |kv| {
                // Free all allocated keys
                allocator.free(kv.key);
                // Free allocated string values
                if (kv.value == .string_) {
                    allocator.free(kv.value.string_);
                }
            }
            allocator.free(row);
        }
        all_rows.deinit(allocator);
    }

    // Parse each line
    var lines = std.mem.splitScalar(u8, data, '\n');
    var row_count: usize = 0;

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;

        // Parse JSON object
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch {
            continue; // Skip invalid lines
        };
        defer parsed.deinit();

        // Flatten object
        var kv_list = std.ArrayList(KeyValue){};
        try flattenObject(allocator, parsed.value, "", &kv_list, 0, config.max_nesting_depth);

        // Update schema from this row
        for (kv_list.items) |kv| {
            const val_type = switch (kv.value) {
                .null_ => ColumnType.null_,
                .bool_ => ColumnType.bool_,
                .int_ => ColumnType.int64,
                .float_ => ColumnType.float64,
                .string_ => ColumnType.string,
            };

            const entry = try column_map.getOrPut(kv.key);
            if (!entry.found_existing) {
                entry.value_ptr.* = .{
                    .name = kv.key,
                    .col_type = val_type,
                    .nullable = kv.value == .null_,
                };
                try column_order.append(allocator, kv.key);
            } else {
                // Promote type
                entry.value_ptr.col_type = entry.value_ptr.col_type.promote(val_type);
                if (kv.value == .null_) {
                    entry.value_ptr.nullable = true;
                }
            }
        }

        // Store row
        try all_rows.append(allocator, try kv_list.toOwnedSlice(allocator));
        row_count += 1;
    }

    if (row_count == 0) return JsonError.EmptyInput;

    // Build columns
    var columns = try allocator.alloc(ColumnData, column_order.items.len);
    for (column_order.items, 0..) |col_name, i| {
        const info = column_map.get(col_name).?;
        // Use initOwnedStrings for string columns since JSON strings are duplicated
        if (info.col_type.toCsvType() == .string) {
            columns[i] = try ColumnData.initOwnedStrings(allocator, col_name, .string);
        } else {
            columns[i] = try ColumnData.initOwned(allocator, col_name, info.col_type.toCsvType());
        }
    }

    // Populate columns from rows
    for (all_rows.items) |row| {
        // Create a map for this row's values
        var row_map = std.StringHashMap(JsonValue).init(allocator);
        defer row_map.deinit();
        for (row) |kv| {
            try row_map.put(kv.key, kv.value);
        }

        // Add value to each column
        for (column_order.items, 0..) |col_name, col_idx| {
            const value = row_map.get(col_name) orelse .null_;
            try appendValue(&columns[col_idx], value, allocator);
        }
    }

    return .{ .columns = columns, .row_count = row_count };
}

/// Parse JSON array data
fn parseJsonArray(
    allocator: std.mem.Allocator,
    data: []const u8,
    config: Config,
) !JsonResult {
    // Parse entire JSON
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, data, .{}) catch {
        return JsonError.InvalidJson;
    };
    defer parsed.deinit();

    if (parsed.value != .array) return JsonError.InvalidJson;

    const arr = parsed.value.array;
    if (arr.items.len == 0) return JsonError.EmptyInput;

    var column_map = std.StringHashMap(ColumnInfo).init(allocator);
    defer column_map.deinit();

    var column_order = std.ArrayList([]const u8){};
    defer column_order.deinit(allocator);

    var all_rows = std.ArrayList([]KeyValue){};
    defer {
        for (all_rows.items) |row| {
            for (row) |kv| {
                // Free all allocated keys
                allocator.free(kv.key);
                // Free allocated string values
                if (kv.value == .string_) {
                    allocator.free(kv.value.string_);
                }
            }
            allocator.free(row);
        }
        all_rows.deinit(allocator);
    }

    // Process each object in array
    for (arr.items) |item| {
        var kv_list = std.ArrayList(KeyValue){};
        try flattenObject(allocator, item, "", &kv_list, 0, config.max_nesting_depth);

        // Update schema
        for (kv_list.items) |kv| {
            const val_type = switch (kv.value) {
                .null_ => ColumnType.null_,
                .bool_ => ColumnType.bool_,
                .int_ => ColumnType.int64,
                .float_ => ColumnType.float64,
                .string_ => ColumnType.string,
            };

            const entry = try column_map.getOrPut(kv.key);
            if (!entry.found_existing) {
                entry.value_ptr.* = .{
                    .name = kv.key,
                    .col_type = val_type,
                    .nullable = kv.value == .null_,
                };
                try column_order.append(allocator, kv.key);
            } else {
                entry.value_ptr.col_type = entry.value_ptr.col_type.promote(val_type);
                if (kv.value == .null_) {
                    entry.value_ptr.nullable = true;
                }
            }
        }

        try all_rows.append(allocator, try kv_list.toOwnedSlice(allocator));
    }

    // Build columns
    var columns = try allocator.alloc(ColumnData, column_order.items.len);
    for (column_order.items, 0..) |col_name, i| {
        const info = column_map.get(col_name).?;
        // Use initOwnedStrings for string columns since JSON strings are duplicated
        if (info.col_type.toCsvType() == .string) {
            columns[i] = try ColumnData.initOwnedStrings(allocator, col_name, .string);
        } else {
            columns[i] = try ColumnData.initOwned(allocator, col_name, info.col_type.toCsvType());
        }
    }

    // Populate columns
    for (all_rows.items) |row| {
        var row_map = std.StringHashMap(JsonValue).init(allocator);
        defer row_map.deinit();
        for (row) |kv| {
            try row_map.put(kv.key, kv.value);
        }

        for (column_order.items, 0..) |col_name, col_idx| {
            const value = row_map.get(col_name) orelse .null_;
            try appendValue(&columns[col_idx], value, allocator);
        }
    }

    return .{ .columns = columns, .row_count = arr.items.len };
}

/// Append a JSON value to a column
/// Note: For string columns, values are duplicated to ensure they outlive the JSON parse tree
fn appendValue(col: *ColumnData, value: JsonValue, allocator: std.mem.Allocator) !void {
    switch (col.col_type) {
        .int64 => {
            const v: i64 = switch (value) {
                .int_ => |i| i,
                .float_ => |f| @intFromFloat(f),
                .bool_ => |b| if (b) @as(i64, 1) else 0,
                .string_ => |s| std.fmt.parseInt(i64, s, 10) catch 0,
                .null_ => 0,
            };
            try col.int64_values.append(col.allocator, v);
        },
        .float64 => {
            const v: f64 = switch (value) {
                .float_ => |f| f,
                .int_ => |i| @floatFromInt(i),
                .bool_ => |b| if (b) @as(f64, 1.0) else 0.0,
                .string_ => |s| std.fmt.parseFloat(f64, s) catch 0.0,
                .null_ => 0.0,
            };
            try col.float64_values.append(col.allocator, v);
        },
        .bool_ => {
            const v: bool = switch (value) {
                .bool_ => |b| b,
                .int_ => |i| i != 0,
                .float_ => |f| f != 0.0,
                .string_ => |s| s.len > 0 and !std.mem.eql(u8, s, "false") and !std.mem.eql(u8, s, "0"),
                .null_ => false,
            };
            try col.bool_values.append(col.allocator, v);
        },
        .string => {
            // All string values must be allocated since the column owns them
            const v: []const u8 = switch (value) {
                .string_ => |s| try allocator.dupe(u8, s),
                .null_ => try allocator.dupe(u8, ""),
                .bool_ => |b| try allocator.dupe(u8, if (b) "true" else "false"),
                .int_ => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
                .float_ => |f| try std.fmt.allocPrint(allocator, "{d}", .{f}),
            };
            try col.string_values.append(col.allocator, v);
        },
    }
}

/// Read entire JSON/JSONL into column-oriented data
pub fn readJson(
    allocator: std.mem.Allocator,
    data: []const u8,
    config: Config,
) !JsonResult {
    const fmt = detectFormat(data);

    return switch (fmt) {
        .jsonl => try parseJsonl(allocator, data, config),
        .json_array => try parseJsonArray(allocator, data, config),
        .empty => JsonError.EmptyInput,
        .unknown => JsonError.InvalidJson,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "detect format - JSON array" {
    try std.testing.expectEqual(Format.json_array, detectFormat("[{\"a\": 1}]"));
    try std.testing.expectEqual(Format.json_array, detectFormat("  [{\"a\": 1}]"));
    try std.testing.expectEqual(Format.json_array, detectFormat("\n[{\"a\": 1}]"));
}

test "detect format - JSONL" {
    try std.testing.expectEqual(Format.jsonl, detectFormat("{\"a\": 1}"));
    try std.testing.expectEqual(Format.jsonl, detectFormat("  {\"a\": 1}"));
    try std.testing.expectEqual(Format.jsonl, detectFormat("{\"a\": 1}\n{\"a\": 2}"));
}

test "detect format - empty" {
    try std.testing.expectEqual(Format.empty, detectFormat(""));
    try std.testing.expectEqual(Format.empty, detectFormat("   "));
    try std.testing.expectEqual(Format.empty, detectFormat("\n\n"));
}

test "detect format - unknown" {
    try std.testing.expectEqual(Format.unknown, detectFormat("abc"));
    try std.testing.expectEqual(Format.unknown, detectFormat("123"));
}

test "type promotion" {
    // Same types
    try std.testing.expectEqual(ColumnType.int64, ColumnType.int64.promote(.int64));
    try std.testing.expectEqual(ColumnType.string, ColumnType.string.promote(.string));

    // Null promotion
    try std.testing.expectEqual(ColumnType.int64, ColumnType.null_.promote(.int64));
    try std.testing.expectEqual(ColumnType.int64, ColumnType.int64.promote(.null_));

    // Int + Float → Float
    try std.testing.expectEqual(ColumnType.float64, ColumnType.int64.promote(.float64));
    try std.testing.expectEqual(ColumnType.float64, ColumnType.float64.promote(.int64));

    // Incompatible → String
    try std.testing.expectEqual(ColumnType.string, ColumnType.int64.promote(.string));
    try std.testing.expectEqual(ColumnType.string, ColumnType.bool_.promote(.int64));
}

test "parse simple JSONL" {
    const allocator = std.testing.allocator;
    const data =
        \\{"name": "Alice", "age": 30}
        \\{"name": "Bob", "age": 25}
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);
}

test "parse JSON array" {
    const allocator = std.testing.allocator;
    const data =
        \\[{"id": 1, "value": 10.5}, {"id": 2, "value": 20.5}]
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);
}

test "type inference int" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1}
        \\{"value": 2}
        \\{"value": 3}
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(csv.ColumnType.int64, result.columns[0].col_type);
    try std.testing.expectEqual(@as(usize, 3), result.columns[0].int64_values.items.len);
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].int64_values.items[0]);
}

test "type inference float" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1.5}
        \\{"value": 2.5}
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(csv.ColumnType.float64, result.columns[0].col_type);
}

test "type inference bool" {
    const allocator = std.testing.allocator;
    const data =
        \\{"active": true}
        \\{"active": false}
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(csv.ColumnType.bool_, result.columns[0].col_type);
    try std.testing.expectEqual(true, result.columns[0].bool_values.items[0]);
    try std.testing.expectEqual(false, result.columns[0].bool_values.items[1]);
}

test "type promotion int to float" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1}
        \\{"value": 2.5}
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Should promote to float64
    try std.testing.expectEqual(csv.ColumnType.float64, result.columns[0].col_type);
}

test "mixed types fallback to string" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1}
        \\{"value": "text"}
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Should fallback to string
    try std.testing.expectEqual(csv.ColumnType.string, result.columns[0].col_type);
}

test "nested object flattening" {
    const allocator = std.testing.allocator;
    const data =
        \\{"user": {"name": "Alice", "profile": {"level": 5}}}
    ;

    const result = try readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Should have flattened columns: user.name, user.profile.level
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
}

test "ColumnType.format" {
    try std.testing.expectEqualStrings("int64", ColumnType.int64.format());
    try std.testing.expectEqualStrings("float64", ColumnType.float64.format());
    try std.testing.expectEqualStrings("bool", ColumnType.bool_.format());
    try std.testing.expectEqualStrings("string", ColumnType.string.format());
    try std.testing.expectEqualStrings("null", ColumnType.null_.format());
}

test "Format.format" {
    try std.testing.expectEqualStrings("json_array", Format.json_array.format());
    try std.testing.expectEqualStrings("jsonl", Format.jsonl.format());
}
