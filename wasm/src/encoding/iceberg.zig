//! Apache Iceberg Table Reader
//!
//! Reads Apache Iceberg tables (directory format).
//!
//! Format structure:
//! - metadata/: Metadata directory
//!   - v1.metadata.json, v2.metadata.json, etc.
//! - data/: Data files (Parquet)
//!
//! Reference: https://iceberg.apache.org/spec/

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Iceberg table reader
pub const IcebergReader = struct {
    allocator: Allocator,
    base_path: []const u8,

    // Table metadata
    format_version: u32,
    table_uuid: []const u8,
    num_columns: usize,
    current_snapshot_id: i64,

    // Schema info
    schema_json: []const u8,

    const Self = @This();

    /// Initialize reader from directory path
    pub fn init(allocator: Allocator, path: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .base_path = path,
            .format_version = 0,
            .table_uuid = "",
            .num_columns = 0,
            .current_snapshot_id = -1,
            .schema_json = "",
        };

        try self.parseTable();
        return self;
    }

    /// Initialize from in-memory metadata (for testing)
    pub fn initFromMetadata(allocator: Allocator, metadata: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .base_path = "",
            .format_version = 0,
            .table_uuid = "",
            .num_columns = 0,
            .current_snapshot_id = -1,
            .schema_json = "",
        };

        try self.parseMetadata(metadata);
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.table_uuid.len > 0) {
            self.allocator.free(self.table_uuid);
        }
        if (self.schema_json.len > 0) {
            self.allocator.free(self.schema_json);
        }
    }

    /// Parse Iceberg table from filesystem
    fn parseTable(self: *Self) !void {
        // Find the latest metadata file
        var path_buf: [4096]u8 = undefined;

        // Try v1, v2, etc. until we find one
        var version: u32 = 1;
        while (version <= 10) : (version += 1) {
            const meta_path = std.fmt.bufPrint(&path_buf, "{s}/metadata/v{d}.metadata.json", .{ self.base_path, version }) catch {
                continue;
            };

            const file = std.fs.cwd().openFile(meta_path, .{}) catch {
                continue;
            };
            defer file.close();

            const metadata = file.readToEndAlloc(self.allocator, 1024 * 1024) catch {
                continue;
            };
            defer self.allocator.free(metadata);

            try self.parseMetadata(metadata);
            return;
        }

        return error.IcebergMetadataNotFound;
    }

    /// Parse Iceberg metadata JSON
    fn parseMetadata(self: *Self, metadata: []const u8) !void {
        // Extract format-version
        if (findJsonNumber(metadata, "\"format-version\"")) |ver| {
            self.format_version = @intCast(ver);
        }

        // Extract table-uuid
        if (findJsonString(metadata, "\"table-uuid\"")) |uuid| {
            self.table_uuid = try self.allocator.dupe(u8, uuid);
        }

        // Extract current-snapshot-id
        if (findJsonNumber(metadata, "\"current-snapshot-id\"")) |sid| {
            self.current_snapshot_id = sid;
        }

        // Extract schema and count fields
        if (std.mem.indexOf(u8, metadata, "\"schema\"")) |schema_start| {
            // Find the schema object - look for the next {
            if (std.mem.indexOfPos(u8, metadata, schema_start, "{")) |obj_start| {
                var depth: usize = 1;
                var pos = obj_start + 1;

                // Find matching }
                while (pos < metadata.len and depth > 0) {
                    if (metadata[pos] == '{') {
                        depth += 1;
                    } else if (metadata[pos] == '}') {
                        depth -= 1;
                    }
                    pos += 1;
                }

                if (depth == 0) {
                    const schema = metadata[obj_start..pos];
                    self.schema_json = try self.allocator.dupe(u8, schema);

                    // Count fields
                    self.num_columns = countJsonFields(schema);
                }
            }
        }
    }

    /// Get format version
    pub fn formatVersion(self: *const Self) u32 {
        return self.format_version;
    }

    /// Get number of columns
    pub fn columnCount(self: *const Self) usize {
        return self.num_columns;
    }

    /// Get current snapshot ID
    pub fn snapshotId(self: *const Self) i64 {
        return self.current_snapshot_id;
    }

    /// Check if path is an Iceberg table
    pub fn isValid(path: []const u8) bool {
        var path_buf: [4096]u8 = undefined;
        const meta_path = std.fmt.bufPrint(&path_buf, "{s}/metadata", .{path}) catch {
            return false;
        };

        var dir = std.fs.cwd().openDir(meta_path, .{}) catch {
            return false;
        };
        dir.close();
        return true;
    }
};

/// Find a JSON string value by key
fn findJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    const key_pos = std.mem.indexOf(u8, json, key) orelse return null;

    var pos = key_pos + key.len;
    while (pos < json.len and (json[pos] == ':' or json[pos] == ' ')) {
        pos += 1;
    }

    if (pos >= json.len or json[pos] != '"') return null;
    pos += 1;

    const start = pos;
    while (pos < json.len and json[pos] != '"') {
        pos += 1;
    }

    if (pos > start) {
        return json[start..pos];
    }
    return null;
}

/// Find a JSON number value by key
fn findJsonNumber(json: []const u8, key: []const u8) ?i64 {
    const key_pos = std.mem.indexOf(u8, json, key) orelse return null;

    var pos = key_pos + key.len;
    while (pos < json.len and (json[pos] == ':' or json[pos] == ' ')) {
        pos += 1;
    }

    if (pos >= json.len) return null;

    // Handle negative numbers
    const is_negative = json[pos] == '-';
    if (is_negative) pos += 1;

    if (pos >= json.len or json[pos] < '0' or json[pos] > '9') return null;

    const start = pos;
    while (pos < json.len and json[pos] >= '0' and json[pos] <= '9') {
        pos += 1;
    }

    const num = std.fmt.parseInt(i64, json[start..pos], 10) catch return null;
    return if (is_negative) -num else num;
}

/// Count fields in Iceberg schema JSON
fn countJsonFields(schema: []const u8) usize {
    // Count occurrences of "name": in the fields array
    var count: usize = 0;
    var pos: usize = 0;

    while (std.mem.indexOfPos(u8, schema, pos, "\"name\":")) |found| {
        count += 1;
        pos = found + 7;
    }

    return count;
}

// Tests
const testing = std.testing;

test "iceberg: parse metadata" {
    const allocator = testing.allocator;

    const metadata =
        \\{
        \\  "format-version": 2,
        \\  "table-uuid": "test-uuid",
        \\  "schema": {
        \\    "type": "struct",
        \\    "fields": [
        \\      {"id": 1, "name": "id", "type": "long"},
        \\      {"id": 2, "name": "name", "type": "string"},
        \\      {"id": 3, "name": "value", "type": "double"}
        \\    ]
        \\  },
        \\  "current-snapshot-id": -1
        \\}
    ;

    var reader = try IcebergReader.initFromMetadata(allocator, metadata);
    defer reader.deinit();

    try testing.expectEqual(@as(u32, 2), reader.formatVersion());
    try testing.expectEqual(@as(usize, 3), reader.columnCount());
    try testing.expectEqual(@as(i64, -1), reader.snapshotId());
}

test "iceberg: read simple fixture" {
    const allocator = testing.allocator;

    const path = "tests/fixtures/simple.iceberg";
    if (!IcebergReader.isValid(path)) {
        std.debug.print("Skipping test: Iceberg fixture not found\n", .{});
        return;
    }

    var reader = try IcebergReader.init(allocator, path);
    defer reader.deinit();

    // Should detect format version 2
    try testing.expectEqual(@as(u32, 2), reader.formatVersion());

    // Should detect columns (3: id, name, value)
    try testing.expectEqual(@as(usize, 3), reader.columnCount());
}
