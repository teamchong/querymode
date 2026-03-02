//! Delta Lake Table Reader
//!
//! Reads Delta Lake tables (directory format).
//!
//! Format structure:
//! - _delta_log/: Transaction log directory
//!   - 00000000000000000000.json: First transaction
//!   - 00000000000000000001.json: Second transaction, etc.
//! - *.parquet: Data files
//!
//! Reference: https://github.com/delta-io/delta/blob/master/PROTOCOL.md

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Delta Lake table reader
pub const DeltaReader = struct {
    allocator: Allocator,
    base_path: []const u8,

    // Table metadata
    table_id: []const u8,
    schema_json: []const u8,
    num_columns: usize,
    num_rows: usize,

    // Data files
    data_files: [][]const u8,

    const Self = @This();

    /// Initialize reader from directory path
    pub fn init(allocator: Allocator, path: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .base_path = path,
            .table_id = "",
            .schema_json = "",
            .num_columns = 0,
            .num_rows = 0,
            .data_files = &.{},
        };

        try self.parseTable();
        return self;
    }

    /// Initialize from in-memory data (for testing without filesystem)
    pub fn initFromLog(allocator: Allocator, log_data: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .base_path = "",
            .table_id = "",
            .schema_json = "",
            .num_columns = 0,
            .num_rows = 0,
            .data_files = &.{},
        };

        try self.parseLogEntry(log_data);
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.table_id.len > 0) {
            self.allocator.free(self.table_id);
        }
        if (self.schema_json.len > 0) {
            self.allocator.free(self.schema_json);
        }
        for (self.data_files) |file| {
            self.allocator.free(file);
        }
        if (self.data_files.len > 0) {
            self.allocator.free(self.data_files);
        }
    }

    /// Parse Delta table from filesystem
    fn parseTable(self: *Self) !void {
        // Build path to delta log
        var path_buf: [4096]u8 = undefined;
        const log_path = std.fmt.bufPrint(&path_buf, "{s}/_delta_log/00000000000000000000.json", .{self.base_path}) catch {
            return error.PathTooLong;
        };

        // Read log file
        const file = std.fs.cwd().openFile(log_path, .{}) catch {
            return error.DeltaLogNotFound;
        };
        defer file.close();

        const log_data = file.readToEndAlloc(self.allocator, 1024 * 1024) catch {
            return error.ReadFailed;
        };
        defer self.allocator.free(log_data);

        try self.parseLogEntry(log_data);
    }

    /// Parse Delta log entry (NDJSON format)
    fn parseLogEntry(self: *Self, log_data: []const u8) !void {
        var files = std.ArrayList([]const u8){};
        errdefer {
            for (files.items) |f| self.allocator.free(f);
            files.deinit(self.allocator);
        }

        var total_rows: usize = 0;

        // Parse each line (NDJSON - newline delimited JSON)
        var lines = std.mem.splitScalar(u8, log_data, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;

            // Check for metaData entry
            if (std.mem.indexOf(u8, line, "\"metaData\"")) |_| {
                try self.parseMetaData(line);
            }
            // Check for add entry
            else if (std.mem.indexOf(u8, line, "\"add\"")) |_| {
                const add_result = try self.parseAddEntry(line);
                if (add_result.path) |path| {
                    try files.append(self.allocator, path);
                }
                total_rows += add_result.rows;
            }
        }

        self.data_files = try files.toOwnedSlice(self.allocator);
        self.num_rows = total_rows;
    }

    /// Parse metaData entry
    fn parseMetaData(self: *Self, line: []const u8) !void {
        // Extract table ID
        if (findJsonString(line, "\"id\"")) |id| {
            self.table_id = try self.allocator.dupe(u8, id);
        }

        // Extract schema
        if (findJsonString(line, "\"schemaString\"")) |schema| {
            // Schema is JSON-escaped, unescape it
            var unescaped = std.ArrayList(u8){};
            defer unescaped.deinit(self.allocator);

            var i: usize = 0;
            while (i < schema.len) {
                if (schema[i] == '\\' and i + 1 < schema.len) {
                    switch (schema[i + 1]) {
                        '"' => {
                            try unescaped.append(self.allocator, '"');
                            i += 2;
                        },
                        '\\' => {
                            try unescaped.append(self.allocator, '\\');
                            i += 2;
                        },
                        'n' => {
                            try unescaped.append(self.allocator, '\n');
                            i += 2;
                        },
                        else => {
                            try unescaped.append(self.allocator, schema[i]);
                            i += 1;
                        },
                    }
                } else {
                    try unescaped.append(self.allocator, schema[i]);
                    i += 1;
                }
            }

            self.schema_json = try unescaped.toOwnedSlice(self.allocator);

            // Count fields in schema
            self.num_columns = countSchemaFields(self.schema_json);
        }
    }

    /// Parse add entry and return path and row count
    fn parseAddEntry(self: *Self, line: []const u8) !struct { path: ?[]const u8, rows: usize } {
        var path: ?[]const u8 = null;
        var rows: usize = 0;

        // Extract path
        if (findJsonString(line, "\"path\"")) |p| {
            path = try self.allocator.dupe(u8, p);
        }

        // Extract numRecords from stats (may be escaped as \" in JSON string)
        // Look for both "numRecords" and \"numRecords\"
        const patterns = [_][]const u8{ "\"numRecords\":", "\\\"numRecords\\\":" };
        for (patterns) |pattern| {
            if (std.mem.indexOf(u8, line, pattern)) |num_pos| {
                // Find the number after the pattern
                var pos = num_pos + pattern.len;
                while (pos < line.len and (line[pos] == ' ' or line[pos] == ':')) {
                    pos += 1;
                }
                if (pos < line.len and line[pos] >= '0' and line[pos] <= '9') {
                    var end = pos;
                    while (end < line.len and line[end] >= '0' and line[end] <= '9') {
                        end += 1;
                    }
                    rows = std.fmt.parseInt(usize, line[pos..end], 10) catch 0;
                    break;
                }
            }
        }

        return .{ .path = path, .rows = rows };
    }

    /// Get number of rows
    pub fn rowCount(self: *const Self) usize {
        return self.num_rows;
    }

    /// Get number of columns
    pub fn columnCount(self: *const Self) usize {
        return self.num_columns;
    }

    /// Get number of data files
    pub fn fileCount(self: *const Self) usize {
        return self.data_files.len;
    }

    /// Check if path is a Delta table
    pub fn isValid(path: []const u8) bool {
        var path_buf: [4096]u8 = undefined;
        const log_path = std.fmt.bufPrint(&path_buf, "{s}/_delta_log", .{path}) catch {
            return false;
        };

        // Check if _delta_log directory exists
        var dir = std.fs.cwd().openDir(log_path, .{}) catch {
            return false;
        };
        dir.close();
        return true;
    }
};

/// Find a JSON string value by key
fn findJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    const key_pos = std.mem.indexOf(u8, json, key) orelse return null;

    // Find the colon after key
    var pos = key_pos + key.len;
    while (pos < json.len and (json[pos] == ':' or json[pos] == ' ')) {
        pos += 1;
    }

    if (pos >= json.len or json[pos] != '"') return null;
    pos += 1; // Skip opening quote

    const start = pos;
    while (pos < json.len and json[pos] != '"') {
        if (json[pos] == '\\' and pos + 1 < json.len) {
            pos += 2; // Skip escaped character
        } else {
            pos += 1;
        }
    }

    if (pos > start) {
        return json[start..pos];
    }
    return null;
}

/// Count fields in Delta schema JSON
fn countSchemaFields(schema: []const u8) usize {
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

test "delta: parse log entry" {
    const allocator = testing.allocator;

    const log_entry =
        \\{"protocol":{"minReaderVersion":1,"minWriterVersion":2}}
        \\{"metaData":{"id":"test-id","schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"id\",\"type\":\"long\"},{\"name\":\"name\",\"type\":\"string\"},{\"name\":\"value\",\"type\":\"double\"}]}"}}
        \\{"add":{"path":"data.parquet","stats":"{\"numRecords\":5}"}}
    ;

    var reader = try DeltaReader.initFromLog(allocator, log_entry);
    defer reader.deinit();

    try testing.expectEqual(@as(usize, 3), reader.columnCount());
    try testing.expectEqual(@as(usize, 5), reader.rowCount());
    try testing.expectEqual(@as(usize, 1), reader.fileCount());
}

test "delta: read simple fixture" {
    const allocator = testing.allocator;

    // Check if fixture exists
    const path = "tests/fixtures/simple.delta";
    if (!DeltaReader.isValid(path)) {
        std.debug.print("Skipping test: Delta fixture not found\n", .{});
        return;
    }

    var reader = try DeltaReader.init(allocator, path);
    defer reader.deinit();

    // Should detect columns (3: id, name, value)
    try testing.expectEqual(@as(usize, 3), reader.columnCount());

    // Should detect rows (5)
    try testing.expectEqual(@as(usize, 5), reader.rowCount());

    // Should have 1 data file
    try testing.expectEqual(@as(usize, 1), reader.fileCount());
}
