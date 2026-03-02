//! Apache Avro Container File Reader
//!
//! Reads Apache Avro container files (.avro).
//!
//! Format structure:
//! - Magic: "Obj" + 0x01 (4 bytes)
//! - File metadata (key-value map with schema, codec)
//! - Sync marker (16 bytes)
//! - Data blocks (count + size + data + sync marker)
//!
//! Reference: https://avro.apache.org/docs/current/specification/#object-container-files

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Avro magic bytes
const AVRO_MAGIC = "Obj\x01";
const SYNC_MARKER_SIZE = 16;

/// Avro data types
pub const AvroType = enum {
    null_type,
    boolean,
    int_type,
    long_type,
    float_type,
    double_type,
    bytes,
    string,
    record,
    enum_type,
    array,
    map,
    fixed,
    union_type,
    unknown,
};

/// Compression codecs
pub const Codec = enum {
    null, // No compression
    deflate,
    snappy,
    unknown,
};

/// Field metadata from schema
pub const FieldInfo = struct {
    name: []const u8,
    avro_type: AvroType,
};

/// Avro container file reader
pub const AvroReader = struct {
    allocator: Allocator,
    data: []const u8,

    // File structure
    sync_marker: [SYNC_MARKER_SIZE]u8,
    codec: Codec,
    data_start: usize,

    // Schema info
    num_fields: usize,
    num_rows: usize,

    // Field names and types (parsed from schema) - use optional for proper tracking
    field_names: ?[][]const u8,
    field_types: ?[]AvroType,

    const Self = @This();

    /// Initialize reader from file data
    pub fn init(allocator: Allocator, data: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .data = data,
            .sync_marker = undefined,
            .codec = .null,
            .data_start = 0,
            .num_fields = 0,
            .num_rows = 0,
            .field_names = null,
            .field_types = null,
        };

        try self.parseFile();
        return self;
    }

    pub fn deinit(self: *Self) void {
        // Free field names
        if (self.field_names) |fn_slice| {
            for (fn_slice) |name| {
                self.allocator.free(name);
            }
            self.allocator.free(fn_slice);
        }
        if (self.field_types) |ft_slice| {
            self.allocator.free(ft_slice);
        }
    }

    /// Parse Avro container file structure
    fn parseFile(self: *Self) !void {
        // Validate minimum size and magic
        if (self.data.len < 4) {
            return error.InvalidAvroFile;
        }

        if (!std.mem.eql(u8, self.data[0..4], AVRO_MAGIC)) {
            return error.InvalidAvroMagic;
        }

        var pos: usize = 4;

        // Parse file metadata (map of string -> bytes)
        const meta_result = try self.parseMetadata(pos);
        pos = meta_result.end_pos;

        // Read sync marker
        if (pos + SYNC_MARKER_SIZE > self.data.len) {
            return error.InvalidAvroFile;
        }
        @memcpy(&self.sync_marker, self.data[pos..][0..SYNC_MARKER_SIZE]);
        pos += SYNC_MARKER_SIZE;

        self.data_start = pos;

        // Count rows by parsing data blocks
        try self.countRows();
    }

    /// Parse file metadata map
    fn parseMetadata(self: *Self, start: usize) !struct { end_pos: usize } {
        var pos = start;

        // Read map block count
        const block_count_result = readVarint(self.data, pos) catch return error.InvalidMetadata;
        const block_count_raw = block_count_result.value;
        pos = block_count_result.end_pos;

        // Zigzag decode for signed
        var block_count: i64 = @bitCast(block_count_raw);
        block_count = (block_count >> 1) ^ (-(block_count & 1));

        if (block_count < 0) {
            // Negative means size follows
            const size_result = readVarint(self.data, pos) catch return error.InvalidMetadata;
            pos = size_result.end_pos;
            block_count = -block_count;
        }

        // Parse each key-value pair
        var i: usize = 0;
        while (i < @as(usize, @intCast(block_count))) : (i += 1) {
            // Read key (string)
            const key_len_result = readVarint(self.data, pos) catch return error.InvalidMetadata;
            const key_len = zigzagDecode(key_len_result.value);
            pos = key_len_result.end_pos;

            if (pos + key_len > self.data.len) return error.InvalidMetadata;
            const key = self.data[pos..][0..key_len];
            pos += key_len;

            // Read value (bytes)
            const val_len_result = readVarint(self.data, pos) catch return error.InvalidMetadata;
            const val_len = zigzagDecode(val_len_result.value);
            pos = val_len_result.end_pos;

            if (pos + val_len > self.data.len) return error.InvalidMetadata;
            const value = self.data[pos..][0..val_len];
            pos += val_len;

            // Handle known keys
            if (std.mem.eql(u8, key, "avro.codec")) {
                if (std.mem.eql(u8, value, "null")) {
                    self.codec = .null;
                } else if (std.mem.eql(u8, value, "deflate")) {
                    self.codec = .deflate;
                } else if (std.mem.eql(u8, value, "snappy")) {
                    self.codec = .snappy;
                } else {
                    self.codec = .unknown;
                }
            } else if (std.mem.eql(u8, key, "avro.schema")) {
                try self.parseSchema(value);
            }
        }

        // Read terminating 0
        if (pos < self.data.len) {
            const term_result = readVarint(self.data, pos) catch return error.InvalidMetadata;
            if (term_result.value == 0) {
                pos = term_result.end_pos;
            }
        }

        return .{ .end_pos = pos };
    }

    /// Parse JSON schema to extract field info
    fn parseSchema(self: *Self, schema_json: []const u8) !void {
        // Simple JSON parsing for Avro schema
        // Schema format: {"type": "record", "name": "...", "fields": [...]}

        var field_names = std.ArrayList([]const u8){};
        var field_types = std.ArrayList(AvroType){};
        errdefer {
            for (field_names.items) |name| {
                self.allocator.free(name);
            }
            field_names.deinit(self.allocator);
            field_types.deinit(self.allocator);
        }

        // Find "fields" array
        const fields_start = std.mem.indexOf(u8, schema_json, "\"fields\"");
        if (fields_start == null) return;

        const array_start = std.mem.indexOfPos(u8, schema_json, fields_start.?, "[");
        if (array_start == null) return;

        var pos = array_start.? + 1;

        // Parse each field object
        while (pos < schema_json.len) {
            // Find next field object
            const obj_start = std.mem.indexOfPos(u8, schema_json, pos, "{");
            if (obj_start == null) break;
            pos = obj_start.? + 1;

            // Find "name" field
            const name_key = std.mem.indexOfPos(u8, schema_json, pos, "\"name\"");
            if (name_key == null) break;

            // Find name value
            const name_colon = std.mem.indexOfPos(u8, schema_json, name_key.?, ":");
            if (name_colon == null) break;

            const name_quote1 = std.mem.indexOfPos(u8, schema_json, name_colon.?, "\"");
            if (name_quote1 == null) break;

            const name_quote2 = std.mem.indexOfPos(u8, schema_json, name_quote1.? + 1, "\"");
            if (name_quote2 == null) break;

            const name = schema_json[name_quote1.? + 1 .. name_quote2.?];

            // Find "type" field
            var avro_type: AvroType = .unknown;
            const type_key = std.mem.indexOfPos(u8, schema_json, pos, "\"type\"");
            if (type_key != null) {
                const type_colon = std.mem.indexOfPos(u8, schema_json, type_key.?, ":");
                if (type_colon != null) {
                    const type_quote1 = std.mem.indexOfPos(u8, schema_json, type_colon.?, "\"");
                    if (type_quote1 != null) {
                        const type_quote2 = std.mem.indexOfPos(u8, schema_json, type_quote1.? + 1, "\"");
                        if (type_quote2 != null) {
                            const type_str = schema_json[type_quote1.? + 1 .. type_quote2.?];
                            avro_type = parseAvroType(type_str);
                        }
                    }
                }
            }

            // Store field
            const name_copy = try self.allocator.dupe(u8, name);
            try field_names.append(self.allocator, name_copy);
            try field_types.append(self.allocator, avro_type);

            // Move to next field
            const obj_end = std.mem.indexOfPos(u8, schema_json, pos, "}");
            if (obj_end == null) break;
            pos = obj_end.? + 1;
        }

        const names = try field_names.toOwnedSlice(self.allocator);
        const types = try field_types.toOwnedSlice(self.allocator);
        self.field_names = names;
        self.field_types = types;
        self.num_fields = names.len;
    }

    /// Parse Avro type string
    fn parseAvroType(type_str: []const u8) AvroType {
        if (std.mem.eql(u8, type_str, "null")) return .null_type;
        if (std.mem.eql(u8, type_str, "boolean")) return .boolean;
        if (std.mem.eql(u8, type_str, "int")) return .int_type;
        if (std.mem.eql(u8, type_str, "long")) return .long_type;
        if (std.mem.eql(u8, type_str, "float")) return .float_type;
        if (std.mem.eql(u8, type_str, "double")) return .double_type;
        if (std.mem.eql(u8, type_str, "bytes")) return .bytes;
        if (std.mem.eql(u8, type_str, "string")) return .string;
        if (std.mem.eql(u8, type_str, "record")) return .record;
        if (std.mem.eql(u8, type_str, "enum")) return .enum_type;
        if (std.mem.eql(u8, type_str, "array")) return .array;
        if (std.mem.eql(u8, type_str, "map")) return .map;
        if (std.mem.eql(u8, type_str, "fixed")) return .fixed;
        return .unknown;
    }

    /// Count total rows by parsing data blocks
    fn countRows(self: *Self) !void {
        var pos = self.data_start;
        var total_rows: usize = 0;

        while (pos + SYNC_MARKER_SIZE < self.data.len) {
            // Read object count (zigzag encoded)
            const count_result = readVarint(self.data, pos) catch break;
            const count_raw = count_result.value;
            pos = count_result.end_pos;

            const count = zigzagDecode(count_raw);
            if (count == 0) break; // End of file

            total_rows += count;

            // Read block size (zigzag encoded)
            const size_result = readVarint(self.data, pos) catch break;
            const block_size = zigzagDecode(size_result.value);
            pos = size_result.end_pos;

            // Skip block data
            if (pos + block_size > self.data.len) break;
            pos += block_size;

            // Skip sync marker
            if (pos + SYNC_MARKER_SIZE > self.data.len) break;
            pos += SYNC_MARKER_SIZE;
        }

        self.num_rows = total_rows;
    }

    /// Get number of columns (fields)
    pub fn columnCount(self: *const Self) usize {
        return self.num_fields;
    }

    /// Get number of rows
    pub fn rowCount(self: *const Self) usize {
        return self.num_rows;
    }

    /// Get codec used
    pub fn getCodec(self: *const Self) Codec {
        return self.codec;
    }

    /// Get field name by index
    pub fn getFieldName(self: *const Self, idx: usize) ?[]const u8 {
        const fn_slice = self.field_names orelse return null;
        if (idx >= fn_slice.len) return null;
        return fn_slice[idx];
    }

    /// Get field type by index
    pub fn getFieldType(self: *const Self, idx: usize) ?AvroType {
        const ft_slice = self.field_types orelse return null;
        if (idx >= ft_slice.len) return null;
        return ft_slice[idx];
    }

    /// Read long (int64) values from a column
    pub fn readLongColumn(self: *Self, col_idx: usize) ![]i64 {
        var values = std.ArrayList(i64){};
        errdefer values.deinit(self.allocator);

        const ft_slice = self.field_types orelse return values.toOwnedSlice(self.allocator);

        if (self.codec != .null) {
            // Compressed data not yet supported
            return values.toOwnedSlice(self.allocator);
        }

        var pos = self.data_start;

        while (pos + SYNC_MARKER_SIZE < self.data.len) {
            // Read object count
            const count_result = readVarint(self.data, pos) catch break;
            const count = zigzagDecode(count_result.value);
            pos = count_result.end_pos;

            if (count == 0) break;

            // Read block size
            const size_result = readVarint(self.data, pos) catch break;
            const block_size = zigzagDecode(size_result.value);
            pos = size_result.end_pos;

            const block_end = pos + block_size;
            if (block_end > self.data.len) break;

            // Parse records in block
            var i: usize = 0;
            while (i < count and pos < block_end) : (i += 1) {
                // Read each field in order
                var field_idx: usize = 0;
                while (field_idx < self.num_fields and pos < block_end) : (field_idx += 1) {
                    const field_type = ft_slice[field_idx];

                    if (field_idx == col_idx and field_type == .long_type) {
                        const val_result = readVarint(self.data, pos) catch break;
                        const val_raw = val_result.value;
                        pos = val_result.end_pos;
                        const val: i64 = @bitCast(val_raw);
                        const decoded = (val >> 1) ^ (-(val & 1));
                        try values.append(self.allocator, decoded);
                    } else {
                        // Skip field
                        pos = try self.skipField(pos, field_type);
                    }
                }
            }

            pos = block_end;

            // Skip sync marker
            if (pos + SYNC_MARKER_SIZE > self.data.len) break;
            pos += SYNC_MARKER_SIZE;
        }

        return values.toOwnedSlice(self.allocator);
    }

    /// Read double values from a column
    pub fn readDoubleColumn(self: *Self, col_idx: usize) ![]f64 {
        var values = std.ArrayList(f64){};
        errdefer values.deinit(self.allocator);

        const ft_slice = self.field_types orelse return values.toOwnedSlice(self.allocator);

        if (self.codec != .null) {
            return values.toOwnedSlice(self.allocator);
        }

        var pos = self.data_start;

        while (pos + SYNC_MARKER_SIZE < self.data.len) {
            const count_result = readVarint(self.data, pos) catch break;
            const count = zigzagDecode(count_result.value);
            pos = count_result.end_pos;

            if (count == 0) break;

            const size_result = readVarint(self.data, pos) catch break;
            const block_size = zigzagDecode(size_result.value);
            pos = size_result.end_pos;

            const block_end = pos + block_size;
            if (block_end > self.data.len) break;

            var i: usize = 0;
            while (i < count and pos < block_end) : (i += 1) {
                var field_idx: usize = 0;
                while (field_idx < self.num_fields and pos < block_end) : (field_idx += 1) {
                    const field_type = ft_slice[field_idx];

                    if (field_idx == col_idx and field_type == .double_type) {
                        if (pos + 8 > self.data.len) break;
                        const bytes = self.data[pos..][0..8];
                        const val: f64 = @bitCast(std.mem.readInt(u64, bytes, .little));
                        try values.append(self.allocator, val);
                        pos += 8;
                    } else {
                        pos = try self.skipField(pos, field_type);
                    }
                }
            }

            pos = block_end;
            if (pos + SYNC_MARKER_SIZE > self.data.len) break;
            pos += SYNC_MARKER_SIZE;
        }

        return values.toOwnedSlice(self.allocator);
    }

    /// Read string values from a column
    pub fn readStringColumn(self: *Self, col_idx: usize) ![][]const u8 {
        var values = std.ArrayList([]const u8){};
        errdefer {
            for (values.items) |s| {
                self.allocator.free(s);
            }
            values.deinit(self.allocator);
        }

        const ft_slice = self.field_types orelse return values.toOwnedSlice(self.allocator);

        if (self.codec != .null) {
            return values.toOwnedSlice(self.allocator);
        }

        var pos = self.data_start;

        while (pos + SYNC_MARKER_SIZE < self.data.len) {
            const count_result = readVarint(self.data, pos) catch break;
            const count = zigzagDecode(count_result.value);
            pos = count_result.end_pos;

            if (count == 0) break;

            const size_result = readVarint(self.data, pos) catch break;
            const block_size = zigzagDecode(size_result.value);
            pos = size_result.end_pos;

            const block_end = pos + block_size;
            if (block_end > self.data.len) break;

            var i: usize = 0;
            while (i < count and pos < block_end) : (i += 1) {
                var field_idx: usize = 0;
                while (field_idx < self.num_fields and pos < block_end) : (field_idx += 1) {
                    const field_type = ft_slice[field_idx];

                    if (field_idx == col_idx and field_type == .string) {
                        const len_result = readVarint(self.data, pos) catch break;
                        const str_len = zigzagDecode(len_result.value);
                        pos = len_result.end_pos;

                        if (pos + str_len > self.data.len) break;
                        const str = try self.allocator.dupe(u8, self.data[pos..][0..str_len]);
                        try values.append(self.allocator, str);
                        pos += str_len;
                    } else {
                        pos = try self.skipField(pos, field_type);
                    }
                }
            }

            pos = block_end;
            if (pos + SYNC_MARKER_SIZE > self.data.len) break;
            pos += SYNC_MARKER_SIZE;
        }

        return values.toOwnedSlice(self.allocator);
    }

    /// Skip a field value
    fn skipField(self: *const Self, pos: usize, field_type: AvroType) !usize {
        var new_pos = pos;

        switch (field_type) {
            .null_type => {},
            .boolean => new_pos += 1,
            .int_type, .long_type => {
                const result = readVarint(self.data, new_pos) catch return error.InvalidData;
                new_pos = result.end_pos;
            },
            .float_type => new_pos += 4,
            .double_type => new_pos += 8,
            .bytes, .string => {
                const result = readVarint(self.data, new_pos) catch return error.InvalidData;
                const len = zigzagDecode(result.value);
                new_pos = result.end_pos + len;
            },
            else => return error.UnsupportedType,
        }

        return new_pos;
    }

    /// Check if file is valid Avro container
    pub fn isValid(data: []const u8) bool {
        if (data.len < 4) return false;
        return std.mem.eql(u8, data[0..4], AVRO_MAGIC);
    }
};

/// Read variable-length integer (Avro uses unsigned varints)
fn readVarint(data: []const u8, start: usize) !struct { value: u64, end_pos: usize } {
    var result: u64 = 0;
    var shift: u6 = 0;
    var pos = start;

    while (pos < data.len) {
        const byte = data[pos];
        pos += 1;

        result |= @as(u64, byte & 0x7F) << shift;

        if (byte & 0x80 == 0) {
            return .{ .value = result, .end_pos = pos };
        }

        shift +|= 7;
        if (shift > 63) return error.VarintTooLong;
    }

    return error.UnexpectedEof;
}

/// Zigzag decode a varint to get signed value
fn zigzagDecode(val: u64) usize {
    const signed: i64 = @bitCast(val);
    const decoded = (signed >> 1) ^ (-(signed & 1));
    if (decoded < 0) return 0;
    return @intCast(decoded);
}

// Tests
const testing = std.testing;

test "avro: magic validation" {
    const allocator = testing.allocator;

    // Invalid magic
    const bad_data = "NOTAVRO";
    const result = AvroReader.init(allocator, bad_data);
    try testing.expectError(error.InvalidAvroMagic, result);
}

test "avro: isValid check" {
    // Valid Avro magic
    const valid = "Obj\x01data";
    try testing.expect(AvroReader.isValid(valid));

    // Invalid
    try testing.expect(!AvroReader.isValid("not avro"));
}

test "avro: read simple fixture" {
    const allocator = testing.allocator;

    // Read fixture file
    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 10 * 1024);
    defer allocator.free(data);

    var reader = try AvroReader.init(allocator, data);
    defer reader.deinit();

    // Should detect 3 columns (id, name, value)
    try testing.expectEqual(@as(usize, 3), reader.columnCount());

    // Should detect 5 rows
    try testing.expectEqual(@as(usize, 5), reader.rowCount());

    // Check codec
    try testing.expectEqual(Codec.null, reader.getCodec());

    // Check field names
    try testing.expectEqualStrings("id", reader.getFieldName(0).?);
    try testing.expectEqualStrings("name", reader.getFieldName(1).?);
    try testing.expectEqualStrings("value", reader.getFieldName(2).?);

    // Check field types
    try testing.expectEqual(AvroType.long_type, reader.getFieldType(0).?);
    try testing.expectEqual(AvroType.string, reader.getFieldType(1).?);
    try testing.expectEqual(AvroType.double_type, reader.getFieldType(2).?);
}

test "avro: read long column from fixture" {
    const allocator = testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 10 * 1024);
    defer allocator.free(data);

    var reader = try AvroReader.init(allocator, data);
    defer reader.deinit();

    // Read first column (id: long)
    const values = try reader.readLongColumn(0);
    defer allocator.free(values);

    // Should have 5 values: [1, 2, 3, 4, 5]
    try testing.expectEqual(@as(usize, 5), values.len);
    if (values.len == 5) {
        try testing.expectEqual(@as(i64, 1), values[0]);
        try testing.expectEqual(@as(i64, 2), values[1]);
        try testing.expectEqual(@as(i64, 3), values[2]);
        try testing.expectEqual(@as(i64, 4), values[3]);
        try testing.expectEqual(@as(i64, 5), values[4]);
    }
}

test "avro: read string column from fixture" {
    const allocator = testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 10 * 1024);
    defer allocator.free(data);

    var reader = try AvroReader.init(allocator, data);
    defer reader.deinit();

    // Read second column (name: string)
    const values = try reader.readStringColumn(1);
    defer {
        for (values) |s| {
            allocator.free(s);
        }
        allocator.free(values);
    }

    // Should have 5 values: ["alice", "bob", "charlie", "diana", "eve"]
    try testing.expectEqual(@as(usize, 5), values.len);
    if (values.len == 5) {
        try testing.expectEqualStrings("alice", values[0]);
        try testing.expectEqualStrings("bob", values[1]);
        try testing.expectEqualStrings("charlie", values[2]);
        try testing.expectEqualStrings("diana", values[3]);
        try testing.expectEqualStrings("eve", values[4]);
    }
}

test "avro: read double column from fixture" {
    const allocator = testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 10 * 1024);
    defer allocator.free(data);

    var reader = try AvroReader.init(allocator, data);
    defer reader.deinit();

    // Read third column (value: double)
    const values = try reader.readDoubleColumn(2);
    defer allocator.free(values);

    // Should have 5 values: [1.1, 2.2, 3.3, 4.4, 5.5]
    try testing.expectEqual(@as(usize, 5), values.len);
    if (values.len == 5) {
        try testing.expectApproxEqAbs(@as(f64, 1.1), values[0], 0.001);
        try testing.expectApproxEqAbs(@as(f64, 2.2), values[1], 0.001);
        try testing.expectApproxEqAbs(@as(f64, 3.3), values[2], 0.001);
        try testing.expectApproxEqAbs(@as(f64, 4.4), values[3], 0.001);
        try testing.expectApproxEqAbs(@as(f64, 5.5), values[4], 0.001);
    }
}
