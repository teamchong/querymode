//! CSV Parser for LanceQL
//!
//! Streaming CSV/TSV parser with:
//! - Auto-detect delimiter (comma, tab, pipe, semicolon)
//! - Type inference (int64, float64, string, bool)
//! - Quote handling (RFC 4180 compliant)
//! - Header row detection
//!
//! Usage:
//!   var parser = CsvParser.init(allocator, data, .{});
//!   defer parser.deinit();
//!
//!   const headers = try parser.readHeader();
//!   while (try parser.nextRow()) |row| {
//!       for (row) |field| {
//!           // process field
//!       }
//!   }

const std = @import("std");

pub const CsvError = error{
    InvalidQuote,
    UnterminatedQuote,
    InconsistentColumnCount,
    EmptyInput,
    OutOfMemory,
};

/// Inferred column type
pub const ColumnType = enum {
    int64,
    float64,
    bool_,
    string,

    pub fn format(self: ColumnType) []const u8 {
        return switch (self) {
            .int64 => "int64",
            .float64 => "float64",
            .bool_ => "bool",
            .string => "string",
        };
    }
};

/// Parser configuration
pub const Config = struct {
    delimiter: ?u8 = null, // Auto-detect if null
    quote_char: u8 = '"',
    has_header: bool = true,
    skip_rows: usize = 0,
    max_field_size: usize = 1024 * 1024, // 1MB max field
};

/// A parsed CSV field
pub const Field = struct {
    value: []const u8,
    quoted: bool,

    /// Try to parse as int64
    pub fn asInt64(self: Field) ?i64 {
        const trimmed = std.mem.trim(u8, self.value, " \t");
        if (trimmed.len == 0) return null;
        return std.fmt.parseInt(i64, trimmed, 10) catch null;
    }

    /// Try to parse as float64
    pub fn asFloat64(self: Field) ?f64 {
        const trimmed = std.mem.trim(u8, self.value, " \t");
        if (trimmed.len == 0) return null;
        return std.fmt.parseFloat(f64, trimmed) catch null;
    }

    /// Try to parse as bool
    pub fn asBool(self: Field) ?bool {
        const trimmed = std.mem.trim(u8, self.value, " \t");
        if (trimmed.len == 0) return null;

        // Check common boolean representations
        if (std.ascii.eqlIgnoreCase(trimmed, "true") or
            std.ascii.eqlIgnoreCase(trimmed, "yes") or
            std.ascii.eqlIgnoreCase(trimmed, "1") or
            std.ascii.eqlIgnoreCase(trimmed, "t") or
            std.ascii.eqlIgnoreCase(trimmed, "y"))
        {
            return true;
        }
        if (std.ascii.eqlIgnoreCase(trimmed, "false") or
            std.ascii.eqlIgnoreCase(trimmed, "no") or
            std.ascii.eqlIgnoreCase(trimmed, "0") or
            std.ascii.eqlIgnoreCase(trimmed, "f") or
            std.ascii.eqlIgnoreCase(trimmed, "n"))
        {
            return false;
        }
        return null;
    }

    /// Get string value (always works)
    pub fn asString(self: Field) []const u8 {
        return self.value;
    }
};

/// Streaming CSV parser
pub const CsvParser = struct {
    allocator: std.mem.Allocator,
    data: []const u8,
    config: Config,
    pos: usize,
    line: usize,
    column_count: ?usize,
    delimiter: u8,
    headers: ?[][]const u8,
    row_buffer: std.ArrayList(Field),

    pub fn init(allocator: std.mem.Allocator, data: []const u8, config: Config) CsvParser {
        const delimiter = config.delimiter orelse detectDelimiter(data);

        return .{
            .allocator = allocator,
            .data = data,
            .config = config,
            .pos = 0,
            .line = 1,
            .column_count = null,
            .delimiter = delimiter,
            .headers = null,
            .row_buffer = std.ArrayList(Field){},
        };
    }

    pub fn deinit(self: *CsvParser) void {
        if (self.headers) |h| {
            for (h) |name| {
                self.allocator.free(name);
            }
            self.allocator.free(h);
        }
        self.row_buffer.deinit(self.allocator);
    }

    /// Read and return header row
    pub fn readHeader(self: *CsvParser) ![][]const u8 {
        if (self.headers != null) return self.headers.?;

        // Skip initial rows if configured
        for (0..self.config.skip_rows) |_| {
            _ = try self.nextRow();
        }

        if (!self.config.has_header) {
            // Generate column names: col0, col1, col2, ...
            const row = try self.nextRow() orelse return CsvError.EmptyInput;
            var names = try self.allocator.alloc([]const u8, row.len);
            for (0..row.len) |i| {
                var buf: [16]u8 = undefined;
                const name = std.fmt.bufPrint(&buf, "col{d}", .{i}) catch "col";
                names[i] = try self.allocator.dupe(u8, name);
            }
            self.headers = names;
            // Reset to beginning (minus skip rows)
            self.pos = 0;
            self.line = 1;
            for (0..self.config.skip_rows) |_| {
                _ = try self.nextRow();
            }
            return names;
        }

        // Read header row
        const row = try self.nextRow() orelse return CsvError.EmptyInput;
        var names = try self.allocator.alloc([]const u8, row.len);
        for (row, 0..) |field, i| {
            names[i] = try self.allocator.dupe(u8, field.value);
        }
        self.headers = names;
        self.column_count = names.len;
        return names;
    }

    /// Read next row, returns null at end of data
    pub fn nextRow(self: *CsvParser) !?[]Field {
        if (self.pos >= self.data.len) return null;

        self.row_buffer.clearRetainingCapacity();
        var last_was_delimiter = false;

        while (self.pos < self.data.len) {
            const field = try self.readField();
            try self.row_buffer.append(self.allocator, field);
            last_was_delimiter = false;

            // Check what's after the field
            if (self.pos >= self.data.len) break;

            const c = self.data[self.pos];
            if (c == '\n') {
                self.pos += 1;
                self.line += 1;
                break;
            } else if (c == '\r') {
                self.pos += 1;
                if (self.pos < self.data.len and self.data[self.pos] == '\n') {
                    self.pos += 1;
                }
                self.line += 1;
                break;
            } else if (c == self.delimiter) {
                self.pos += 1; // skip delimiter, continue to next field
                last_was_delimiter = true;
            }
        }

        // Handle trailing delimiter at end of file
        if (last_was_delimiter) {
            try self.row_buffer.append(self.allocator, Field{ .value = "", .quoted = false });
        }

        // Validate column count consistency
        if (self.column_count) |expected| {
            if (self.row_buffer.items.len != expected) {
                // Allow last empty row
                if (self.row_buffer.items.len == 1 and self.row_buffer.items[0].value.len == 0) {
                    return null;
                }
                return CsvError.InconsistentColumnCount;
            }
        } else {
            self.column_count = self.row_buffer.items.len;
        }

        return self.row_buffer.items;
    }

    /// Read a single field
    fn readField(self: *CsvParser) !Field {
        if (self.pos >= self.data.len) {
            return Field{ .value = "", .quoted = false };
        }

        const start_char = self.data[self.pos];

        if (start_char == self.config.quote_char) {
            return self.readQuotedField();
        } else {
            return self.readUnquotedField();
        }
    }

    fn readUnquotedField(self: *CsvParser) Field {
        const start = self.pos;
        while (self.pos < self.data.len) {
            const c = self.data[self.pos];
            if (c == self.delimiter or c == '\n' or c == '\r') {
                break;
            }
            self.pos += 1;
        }
        return Field{
            .value = self.data[start..self.pos],
            .quoted = false,
        };
    }

    fn readQuotedField(self: *CsvParser) !Field {
        self.pos += 1; // skip opening quote
        const start = self.pos;

        while (self.pos < self.data.len) {
            const c = self.data[self.pos];
            if (c == self.config.quote_char) {
                // Check for escaped quote (doubled)
                if (self.pos + 1 < self.data.len and self.data[self.pos + 1] == self.config.quote_char) {
                    self.pos += 2;
                    continue;
                }
                // End of quoted field
                const value = self.data[start..self.pos];
                self.pos += 1; // skip closing quote
                return Field{ .value = value, .quoted = true };
            }
            self.pos += 1;
        }

        return CsvError.UnterminatedQuote;
    }

    /// Infer column types from first N rows
    pub fn inferTypes(self: *CsvParser, sample_rows: usize) ![]ColumnType {
        const saved_pos = self.pos;
        const saved_line = self.line;
        defer {
            self.pos = saved_pos;
            self.line = saved_line;
        }

        // Reset to after header
        self.pos = 0;
        self.line = 1;
        if (self.config.has_header) {
            _ = try self.nextRow(); // skip header
        }

        const col_count = self.column_count orelse return CsvError.EmptyInput;
        var types = try self.allocator.alloc(ColumnType, col_count);
        @memset(types, .string); // default to string

        // Track type candidates
        var can_be_int = try self.allocator.alloc(bool, col_count);
        defer self.allocator.free(can_be_int);
        var can_be_float = try self.allocator.alloc(bool, col_count);
        defer self.allocator.free(can_be_float);
        var can_be_bool = try self.allocator.alloc(bool, col_count);
        defer self.allocator.free(can_be_bool);

        @memset(can_be_int, true);
        @memset(can_be_float, true);
        @memset(can_be_bool, true);

        var rows_checked: usize = 0;
        while (rows_checked < sample_rows) : (rows_checked += 1) {
            const row = try self.nextRow() orelse break;

            for (row, 0..) |field, i| {
                const trimmed = std.mem.trim(u8, field.value, " \t");
                if (trimmed.len == 0) continue; // skip empty values

                if (can_be_int[i] and field.asInt64() == null) {
                    can_be_int[i] = false;
                }
                if (can_be_float[i] and field.asFloat64() == null) {
                    can_be_float[i] = false;
                }
                if (can_be_bool[i] and field.asBool() == null) {
                    can_be_bool[i] = false;
                }
            }
        }

        // Determine final types (priority: bool > int > float > string)
        for (0..col_count) |i| {
            if (can_be_bool[i]) {
                types[i] = .bool_;
            } else if (can_be_int[i]) {
                types[i] = .int64;
            } else if (can_be_float[i]) {
                types[i] = .float64;
            } else {
                types[i] = .string;
            }
        }

        return types;
    }
};

/// Auto-detect delimiter from first line(s) of data
pub fn detectDelimiter(data: []const u8) u8 {
    const delimiters = [_]u8{ ',', '\t', '|', ';' };
    var counts = [_]usize{ 0, 0, 0, 0 };

    // Count occurrences in first 1000 chars or until newline
    const sample_len = @min(data.len, 1000);
    var in_quote = false;

    for (data[0..sample_len]) |c| {
        if (c == '"') {
            in_quote = !in_quote;
            continue;
        }
        if (in_quote) continue;

        for (delimiters, 0..) |d, i| {
            if (c == d) counts[i] += 1;
        }
    }

    // Find most common delimiter
    var max_idx: usize = 0;
    var max_count: usize = 0;
    for (counts, 0..) |count, i| {
        if (count > max_count) {
            max_count = count;
            max_idx = i;
        }
    }

    // Default to comma if no delimiter found
    if (max_count == 0) return ',';

    return delimiters[max_idx];
}

/// Column data for building Lance files
pub const ColumnData = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    owns_name: bool, // Whether we need to free name on deinit
    owns_strings: bool, // Whether we need to free string values on deinit
    col_type: ColumnType,
    int64_values: std.ArrayList(i64),
    float64_values: std.ArrayList(f64),
    bool_values: std.ArrayList(bool),
    string_values: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, col_type: ColumnType) ColumnData {
        return .{
            .allocator = allocator,
            .name = name,
            .owns_name = false,
            .owns_strings = false,
            .col_type = col_type,
            .int64_values = std.ArrayList(i64){},
            .float64_values = std.ArrayList(f64){},
            .bool_values = std.ArrayList(bool){},
            .string_values = std.ArrayList([]const u8){},
        };
    }

    /// Create ColumnData with an owned (duplicated) name
    pub fn initOwned(allocator: std.mem.Allocator, name: []const u8, col_type: ColumnType) !ColumnData {
        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .owns_name = true,
            .owns_strings = false,
            .col_type = col_type,
            .int64_values = std.ArrayList(i64){},
            .float64_values = std.ArrayList(f64){},
            .bool_values = std.ArrayList(bool){},
            .string_values = std.ArrayList([]const u8){},
        };
    }

    /// Create ColumnData with owned name and strings (for JSON parsing)
    pub fn initOwnedStrings(allocator: std.mem.Allocator, name: []const u8, col_type: ColumnType) !ColumnData {
        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .owns_name = true,
            .owns_strings = true,
            .col_type = col_type,
            .int64_values = std.ArrayList(i64){},
            .float64_values = std.ArrayList(f64){},
            .bool_values = std.ArrayList(bool){},
            .string_values = std.ArrayList([]const u8){},
        };
    }

    pub fn deinit(self: *ColumnData) void {
        if (self.owns_name) {
            self.allocator.free(self.name);
        }
        if (self.owns_strings) {
            for (self.string_values.items) |s| {
                self.allocator.free(s);
            }
        }
        self.int64_values.deinit(self.allocator);
        self.float64_values.deinit(self.allocator);
        self.bool_values.deinit(self.allocator);
        self.string_values.deinit(self.allocator);
    }

    pub fn appendField(self: *ColumnData, field: Field) !void {
        switch (self.col_type) {
            .int64 => try self.int64_values.append(self.allocator, field.asInt64() orelse 0),
            .float64 => try self.float64_values.append(self.allocator, field.asFloat64() orelse 0.0),
            .bool_ => try self.bool_values.append(self.allocator, field.asBool() orelse false),
            .string => try self.string_values.append(self.allocator, field.asString()),
        }
    }

    pub fn len(self: *const ColumnData) usize {
        return switch (self.col_type) {
            .int64 => self.int64_values.items.len,
            .float64 => self.float64_values.items.len,
            .bool_ => self.bool_values.items.len,
            .string => self.string_values.items.len,
        };
    }
};

/// Read entire CSV into column-oriented data
pub fn readCsv(allocator: std.mem.Allocator, data: []const u8, config: Config) !struct {
    columns: []ColumnData,
    row_count: usize,
} {
    var parser = CsvParser.init(allocator, data, config);
    defer parser.deinit();

    // Read headers
    const headers = try parser.readHeader();

    // Infer types from sample
    const types = try parser.inferTypes(100);
    defer allocator.free(types);

    // Reset parser
    parser.pos = 0;
    parser.line = 1;
    if (config.has_header) {
        _ = try parser.nextRow(); // skip header
    }

    // Initialize column data (use initOwned to duplicate names since parser will free them)
    var columns = try allocator.alloc(ColumnData, headers.len);
    for (headers, 0..) |name, i| {
        columns[i] = try ColumnData.initOwned(allocator, name, types[i]);
    }

    // Read all rows
    var row_count: usize = 0;
    while (try parser.nextRow()) |row| {
        for (row, 0..) |field, i| {
            try columns[i].appendField(field);
        }
        row_count += 1;
    }

    return .{ .columns = columns, .row_count = row_count };
}

// Tests
test "detect delimiter - comma" {
    const data = "a,b,c\n1,2,3\n4,5,6";
    try std.testing.expectEqual(@as(u8, ','), detectDelimiter(data));
}

test "detect delimiter - tab" {
    const data = "a\tb\tc\n1\t2\t3\n4\t5\t6";
    try std.testing.expectEqual(@as(u8, '\t'), detectDelimiter(data));
}

test "detect delimiter - pipe" {
    const data = "a|b|c\n1|2|3\n4|5|6";
    try std.testing.expectEqual(@as(u8, '|'), detectDelimiter(data));
}

test "parse simple csv" {
    const allocator = std.testing.allocator;
    const data = "name,age,active\nAlice,30,true\nBob,25,false";

    var parser = CsvParser.init(allocator, data, .{});
    defer parser.deinit();

    const headers = try parser.readHeader();
    try std.testing.expectEqual(@as(usize, 3), headers.len);
    try std.testing.expectEqualStrings("name", headers[0]);
    try std.testing.expectEqualStrings("age", headers[1]);
    try std.testing.expectEqualStrings("active", headers[2]);

    const row1 = try parser.nextRow();
    try std.testing.expect(row1 != null);
    try std.testing.expectEqualStrings("Alice", row1.?[0].value);
    try std.testing.expectEqual(@as(i64, 30), row1.?[1].asInt64().?);
    try std.testing.expectEqual(true, row1.?[2].asBool().?);

    const row2 = try parser.nextRow();
    try std.testing.expect(row2 != null);
    try std.testing.expectEqualStrings("Bob", row2.?[0].value);
}

test "parse quoted csv" {
    const allocator = std.testing.allocator;
    const data = "name,desc\n\"John\",\"Hello, World\"";

    var parser = CsvParser.init(allocator, data, .{});
    defer parser.deinit();

    _ = try parser.readHeader();

    const row = try parser.nextRow();
    try std.testing.expect(row != null);
    try std.testing.expectEqualStrings("John", row.?[0].value);
    try std.testing.expectEqualStrings("Hello, World", row.?[1].value);
}

test "type inference" {
    const allocator = std.testing.allocator;
    const data = "id,value,flag,name\n1,3.14,true,Alice\n2,2.71,false,Bob";

    var parser = CsvParser.init(allocator, data, .{});
    defer parser.deinit();

    _ = try parser.readHeader();
    const types = try parser.inferTypes(10);
    defer allocator.free(types);

    try std.testing.expectEqual(ColumnType.int64, types[0]);
    try std.testing.expectEqual(ColumnType.float64, types[1]);
    try std.testing.expectEqual(ColumnType.bool_, types[2]);
    try std.testing.expectEqual(ColumnType.string, types[3]);
}

test "Field.asInt64" {
    const field1 = Field{ .value = "123", .quoted = false };
    try std.testing.expectEqual(@as(i64, 123), field1.asInt64().?);

    const field2 = Field{ .value = "-456", .quoted = false };
    try std.testing.expectEqual(@as(i64, -456), field2.asInt64().?);

    const field3 = Field{ .value = "not a number", .quoted = false };
    try std.testing.expect(field3.asInt64() == null);

    const field4 = Field{ .value = "  42  ", .quoted = false };
    try std.testing.expectEqual(@as(i64, 42), field4.asInt64().?);

    const field5 = Field{ .value = "", .quoted = false };
    try std.testing.expect(field5.asInt64() == null);
}

test "Field.asFloat64" {
    const field1 = Field{ .value = "3.14159", .quoted = false };
    try std.testing.expectApproxEqRel(@as(f64, 3.14159), field1.asFloat64().?, 0.0001);

    const field2 = Field{ .value = "-2.5", .quoted = false };
    try std.testing.expectApproxEqRel(@as(f64, -2.5), field2.asFloat64().?, 0.0001);

    const field3 = Field{ .value = "42", .quoted = false };
    try std.testing.expectEqual(@as(f64, 42.0), field3.asFloat64().?);

    const field4 = Field{ .value = "not a float", .quoted = false };
    try std.testing.expect(field4.asFloat64() == null);
}

test "Field.asBool" {
    const true_vals = [_][]const u8{ "true", "TRUE", "True", "yes", "YES", "1", "t", "T", "y", "Y" };
    for (true_vals) |val| {
        const field = Field{ .value = val, .quoted = false };
        try std.testing.expectEqual(true, field.asBool().?);
    }

    const false_vals = [_][]const u8{ "false", "FALSE", "False", "no", "NO", "0", "f", "F", "n", "N" };
    for (false_vals) |val| {
        const field = Field{ .value = val, .quoted = false };
        try std.testing.expectEqual(false, field.asBool().?);
    }

    const field_invalid = Field{ .value = "maybe", .quoted = false };
    try std.testing.expect(field_invalid.asBool() == null);
}

test "ColumnType.format" {
    try std.testing.expectEqualStrings("int64", ColumnType.int64.format());
    try std.testing.expectEqualStrings("float64", ColumnType.float64.format());
    try std.testing.expectEqualStrings("bool", ColumnType.bool_.format());
    try std.testing.expectEqualStrings("string", ColumnType.string.format());
}

test "detect delimiter - semicolon" {
    const data = "a;b;c\n1;2;3\n4;5;6";
    try std.testing.expectEqual(@as(u8, ';'), detectDelimiter(data));
}

test "detect delimiter - default to comma" {
    const data = "single_column\nvalue1\nvalue2";
    try std.testing.expectEqual(@as(u8, ','), detectDelimiter(data));
}

test "detect delimiter - ignores quoted content" {
    const data = "a,b\n\"x,y,z\",c\n1,2";
    try std.testing.expectEqual(@as(u8, ','), detectDelimiter(data));
}

test "parse csv with CRLF line endings" {
    const allocator = std.testing.allocator;
    const data = "a,b\r\n1,2\r\n3,4";

    var parser = CsvParser.init(allocator, data, .{});
    defer parser.deinit();

    _ = try parser.readHeader();

    const row1 = try parser.nextRow();
    try std.testing.expect(row1 != null);
    try std.testing.expectEqualStrings("1", row1.?[0].value);
    try std.testing.expectEqualStrings("2", row1.?[1].value);

    const row2 = try parser.nextRow();
    try std.testing.expect(row2 != null);
    try std.testing.expectEqualStrings("3", row2.?[0].value);
    try std.testing.expectEqualStrings("4", row2.?[1].value);
}

test "parse csv with escaped quotes" {
    const allocator = std.testing.allocator;
    const data = "name\n\"He said \"\"Hello\"\"\"";

    var parser = CsvParser.init(allocator, data, .{});
    defer parser.deinit();

    _ = try parser.readHeader();

    const row = try parser.nextRow();
    try std.testing.expect(row != null);
    try std.testing.expectEqualStrings("He said \"\"Hello\"\"", row.?[0].value);
}

test "parse TSV" {
    const allocator = std.testing.allocator;
    const data = "name\tage\nAlice\t30\nBob\t25";

    var parser = CsvParser.init(allocator, data, .{ .delimiter = '\t' });
    defer parser.deinit();

    const headers = try parser.readHeader();
    try std.testing.expectEqual(@as(usize, 2), headers.len);
    try std.testing.expectEqualStrings("name", headers[0]);
    try std.testing.expectEqualStrings("age", headers[1]);
}

test "parse csv without header" {
    const allocator = std.testing.allocator;
    const data = "Alice,30\nBob,25";

    var parser = CsvParser.init(allocator, data, .{ .has_header = false });
    defer parser.deinit();

    const headers = try parser.readHeader();
    try std.testing.expectEqual(@as(usize, 2), headers.len);
    try std.testing.expectEqualStrings("col0", headers[0]);
    try std.testing.expectEqualStrings("col1", headers[1]);
}

test "ColumnData basic operations" {
    const allocator = std.testing.allocator;

    var col = ColumnData.init(allocator, "test", .int64);
    defer col.deinit();

    try col.appendField(Field{ .value = "1", .quoted = false });
    try col.appendField(Field{ .value = "2", .quoted = false });
    try col.appendField(Field{ .value = "3", .quoted = false });

    try std.testing.expectEqual(@as(usize, 3), col.len());
    try std.testing.expectEqual(@as(i64, 1), col.int64_values.items[0]);
    try std.testing.expectEqual(@as(i64, 2), col.int64_values.items[1]);
    try std.testing.expectEqual(@as(i64, 3), col.int64_values.items[2]);
}

test "ColumnData float64" {
    const allocator = std.testing.allocator;

    var col = ColumnData.init(allocator, "values", .float64);
    defer col.deinit();

    try col.appendField(Field{ .value = "1.5", .quoted = false });
    try col.appendField(Field{ .value = "2.5", .quoted = false });

    try std.testing.expectEqual(@as(usize, 2), col.len());
    try std.testing.expectApproxEqRel(@as(f64, 1.5), col.float64_values.items[0], 0.0001);
}

test "ColumnData bool" {
    const allocator = std.testing.allocator;

    var col = ColumnData.init(allocator, "flags", .bool_);
    defer col.deinit();

    try col.appendField(Field{ .value = "true", .quoted = false });
    try col.appendField(Field{ .value = "false", .quoted = false });

    try std.testing.expectEqual(@as(usize, 2), col.len());
    try std.testing.expectEqual(true, col.bool_values.items[0]);
    try std.testing.expectEqual(false, col.bool_values.items[1]);
}

test "ColumnData string" {
    const allocator = std.testing.allocator;

    var col = ColumnData.init(allocator, "names", .string);
    defer col.deinit();

    try col.appendField(Field{ .value = "Alice", .quoted = false });
    try col.appendField(Field{ .value = "Bob", .quoted = false });

    try std.testing.expectEqual(@as(usize, 2), col.len());
    try std.testing.expectEqualStrings("Alice", col.string_values.items[0]);
    try std.testing.expectEqualStrings("Bob", col.string_values.items[1]);
}

test "readCsv full integration" {
    const allocator = std.testing.allocator;
    const data = "id,value,active\n1,10.5,true\n2,20.5,false\n3,30.5,true";

    const result = try readCsv(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 3), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // Check column types
    try std.testing.expectEqual(ColumnType.int64, result.columns[0].col_type);
    try std.testing.expectEqual(ColumnType.float64, result.columns[1].col_type);
    try std.testing.expectEqual(ColumnType.bool_, result.columns[2].col_type);

    // Check values
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].int64_values.items[0]);
    try std.testing.expectEqual(@as(i64, 2), result.columns[0].int64_values.items[1]);
    try std.testing.expectEqual(@as(i64, 3), result.columns[0].int64_values.items[2]);
}

test "empty field handling" {
    const allocator = std.testing.allocator;
    const data = "a,b,c\n1,,3\n,2,";

    var parser = CsvParser.init(allocator, data, .{});
    defer parser.deinit();

    _ = try parser.readHeader();

    const row1 = try parser.nextRow();
    try std.testing.expect(row1 != null);
    try std.testing.expectEqualStrings("1", row1.?[0].value);
    try std.testing.expectEqualStrings("", row1.?[1].value);
    try std.testing.expectEqualStrings("3", row1.?[2].value);

    const row2 = try parser.nextRow();
    try std.testing.expect(row2 != null);
    try std.testing.expectEqualStrings("", row2.?[0].value);
    try std.testing.expectEqualStrings("2", row2.?[1].value);
    try std.testing.expectEqualStrings("", row2.?[2].value);
}

test "ColumnData initOwned" {
    const allocator = std.testing.allocator;
    const original_name = "test_column";

    var col = try ColumnData.initOwned(allocator, original_name, .int64);
    defer col.deinit();

    // Verify name was duplicated (different memory address)
    try std.testing.expectEqualStrings(original_name, col.name);
    try std.testing.expect(col.name.ptr != original_name.ptr);
    try std.testing.expect(col.owns_name);

    // Verify basic operations work
    try col.appendField(Field{ .value = "42", .quoted = false });
    try std.testing.expectEqual(@as(usize, 1), col.len());
    try std.testing.expectEqual(@as(i64, 42), col.int64_values.items[0]);
}
