//! Lance manifest writer.
//!
//! Creates `_versions/N.manifest` files for Lance datasets.
//! Manifests contain:
//! - Schema (field definitions)
//! - Fragments (data file references)
//! - Version metadata (number, timestamp, transaction ID)
//!
//! ## File Layout
//! | Offset | Size | Content |
//! |--------|------|---------|
//! | 0 | 4 | Protobuf length (little-endian u32) |
//! | 4 | N | Protobuf-encoded Manifest message |
//! | 4+N | 12 | Footer (reserved padding) |
//! | EOF-4 | 4 | Magic "LANC" |
//!
//! ## Manifest Protobuf Fields
//! | Field | Number | Type |
//! |-------|--------|------|
//! | fields | 1 | repeated Field (schema) |
//! | fragments | 2 | repeated DataFragment |
//! | version | 3 | uint64 |
//! | timestamp | 7 | google.protobuf.Timestamp |
//! | reader_feature_flags | 9 | uint64 |
//! | writer_feature_flags | 10 | uint64 |
//! | transaction_id | 12 | string |
//! | writer_version | 13 | WriterVersion |
//! | next_row_id | 14 | uint64 |

const std = @import("std");
const proto_mod = @import("querymode.proto");

const ProtoEncoder = proto_mod.ProtoEncoder;

/// Magic bytes identifying a Lance manifest
pub const MANIFEST_MAGIC = "LANC";

/// Writer version string
pub const WRITER_VERSION = "querymode-0.1";

/// Errors that can occur during manifest writing
pub const ManifestWriteError = error{
    OutOfMemory,
    NoFragments,
    InvalidField,
};

/// Field type enum matching Lance's field types
pub const FieldType = enum(u32) {
    /// Non-leaf node (struct/list parent)
    parent = 0,
    /// Repeated field (list)
    repeated = 1,
    /// Leaf (data) field
    leaf = 2,
};

/// Schema field definition
pub const Field = struct {
    /// Field type (parent, repeated, leaf)
    field_type: FieldType = .leaf,
    /// Column name
    name: []const u8,
    /// Field ID (unique within schema)
    id: u32,
    /// Parent field ID (-1 for root level)
    parent_id: i32 = -1,
    /// Logical type string (e.g., "int64", "string", "double")
    logical_type: []const u8,
    /// Whether this field allows nulls
    nullable: bool = true,
    /// For fixed-size lists: dimension
    dimension: ?u32 = null,
};

/// Data file reference within a fragment
pub const DataFile = struct {
    /// Relative path to the data file (e.g., "data/uuid.lance")
    path: []const u8,
    /// Field IDs in this file (empty means all fields)
    fields: []const u32 = &[_]u32{},
    /// Column indices this file contains
    column_indices: []const u32 = &[_]u32{},
};

/// Fragment (batch of rows) in the dataset
pub const Fragment = struct {
    /// Fragment ID (monotonically increasing)
    id: u64,
    /// Data files in this fragment
    files: []const DataFile,
    /// Number of physical rows in this fragment
    physical_rows: u64,
    /// Deletion file path (if any)
    deletion_file: ?[]const u8 = null,
};

/// Timestamp (Google protobuf Timestamp format)
pub const Timestamp = struct {
    /// Seconds since Unix epoch
    seconds: i64,
    /// Nanoseconds (0-999,999,999)
    nanos: i32 = 0,

    /// Create timestamp from current time
    pub fn now() Timestamp {
        const ns = std.time.nanoTimestamp();
        const seconds = @divFloor(ns, 1_000_000_000);
        const nanos: i32 = @intCast(@mod(ns, 1_000_000_000));
        return .{ .seconds = seconds, .nanos = nanos };
    }

    /// Create timestamp from milliseconds since epoch
    pub fn fromMillis(millis: i64) Timestamp {
        return .{
            .seconds = @divFloor(millis, 1000),
            .nanos = @intCast(@mod(millis, 1000) * 1_000_000),
        };
    }
};

/// Manifest writer for creating Lance dataset versions
pub const ManifestWriter = struct {
    allocator: std.mem.Allocator,
    fields: std.ArrayListUnmanaged(Field),
    fragments: std.ArrayListUnmanaged(Fragment),
    version: u64,
    timestamp: Timestamp,
    transaction_id: []const u8,
    reader_feature_flags: u64,
    writer_feature_flags: u64,
    next_row_id: u64,

    const Self = @This();

    /// Create a new manifest writer
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .fields = std.ArrayListUnmanaged(Field){},
            .fragments = std.ArrayListUnmanaged(Fragment){},
            .version = 1,
            .timestamp = Timestamp.now(),
            .transaction_id = "",
            .reader_feature_flags = 0,
            .writer_feature_flags = 0,
            .next_row_id = 0,
        };
    }

    /// Free all resources
    pub fn deinit(self: *Self) void {
        self.fields.deinit(self.allocator);
        self.fragments.deinit(self.allocator);
    }

    /// Set the schema fields
    pub fn setSchema(self: *Self, fields: []const Field) !void {
        self.fields.clearRetainingCapacity();
        try self.fields.appendSlice(self.allocator, fields);
    }

    /// Add a schema field
    pub fn addField(self: *Self, field: Field) !void {
        try self.fields.append(self.allocator, field);
    }

    /// Add a fragment
    pub fn addFragment(self: *Self, fragment: Fragment) !void {
        try self.fragments.append(self.allocator, fragment);
        self.next_row_id += fragment.physical_rows;
    }

    /// Set the version number
    pub fn setVersion(self: *Self, version: u64) void {
        self.version = version;
    }

    /// Set the timestamp
    pub fn setTimestamp(self: *Self, timestamp: Timestamp) void {
        self.timestamp = timestamp;
    }

    /// Set the transaction ID
    pub fn setTransactionId(self: *Self, id: []const u8) void {
        self.transaction_id = id;
    }

    /// Encode the manifest to bytes
    pub fn encode(self: *Self) ![]const u8 {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        // Encode schema fields (field 1)
        for (self.fields.items) |field| {
            const field_bytes = try self.encodeField(field);
            defer self.allocator.free(field_bytes);
            try encoder.writeMessageField(1, field_bytes);
        }

        // Encode fragments (field 2)
        for (self.fragments.items) |fragment| {
            const fragment_bytes = try self.encodeFragment(fragment);
            defer self.allocator.free(fragment_bytes);
            try encoder.writeMessageField(2, fragment_bytes);
        }

        // Version (field 3)
        try encoder.writeVarintField(3, self.version);

        // Timestamp (field 7)
        const timestamp_bytes = try self.encodeTimestamp(self.timestamp);
        defer self.allocator.free(timestamp_bytes);
        try encoder.writeMessageField(7, timestamp_bytes);

        // Reader feature flags (field 9)
        if (self.reader_feature_flags > 0) {
            try encoder.writeVarintField(9, self.reader_feature_flags);
        }

        // Writer feature flags (field 10)
        if (self.writer_feature_flags > 0) {
            try encoder.writeVarintField(10, self.writer_feature_flags);
        }

        // Transaction ID (field 12)
        if (self.transaction_id.len > 0) {
            try encoder.writeStringField(12, self.transaction_id);
        }

        // Writer version (field 13)
        const writer_version_bytes = try self.encodeWriterVersion();
        defer self.allocator.free(writer_version_bytes);
        try encoder.writeMessageField(13, writer_version_bytes);

        // Next row ID (field 14)
        try encoder.writeVarintField(14, self.next_row_id);

        // Build final manifest file
        const proto_bytes = encoder.getBytes();
        const proto_len: u32 = @intCast(proto_bytes.len);

        // Total size: 4 (length) + proto + 12 (footer padding) + 4 (magic)
        const total_size = 4 + proto_bytes.len + 12 + 4;
        var output = try self.allocator.alloc(u8, total_size);

        // Write protobuf length
        std.mem.writeInt(u32, output[0..4], proto_len, .little);

        // Write protobuf content
        @memcpy(output[4..][0..proto_bytes.len], proto_bytes);

        // Write footer padding (zeros)
        @memset(output[4 + proto_bytes.len ..][0..12], 0);

        // Write magic
        @memcpy(output[total_size - 4 ..][0..4], MANIFEST_MAGIC);

        return output;
    }

    fn encodeField(self: *Self, field: Field) ![]const u8 {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        // Field 1: type (enum)
        try encoder.writeVarintField(1, @intFromEnum(field.field_type));

        // Field 2: name
        try encoder.writeStringField(2, field.name);

        // Field 3: id
        try encoder.writeVarintField(3, field.id);

        // Field 4: parent_id (as unsigned, -1 becomes 0xFFFFFFFF)
        const parent_u32: u32 = @bitCast(field.parent_id);
        try encoder.writeVarintField(4, parent_u32);

        // Field 5: logical_type
        try encoder.writeStringField(5, field.logical_type);

        // Field 6: nullable
        try encoder.writeVarintField(6, if (field.nullable) 1 else 0);

        // Field 8: dimension (for fixed_size_list)
        if (field.dimension) |dim| {
            try encoder.writeVarintField(8, dim);
        }

        return try self.allocator.dupe(u8, encoder.getBytes());
    }

    fn encodeFragment(self: *Self, fragment: Fragment) ![]const u8 {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        // Field 1: id
        try encoder.writeVarintField(1, fragment.id);

        // Field 2: files (repeated DataFile)
        for (fragment.files) |file| {
            const file_bytes = try self.encodeDataFile(file);
            defer self.allocator.free(file_bytes);
            try encoder.writeMessageField(2, file_bytes);
        }

        // Field 3: deletion_file (optional)
        if (fragment.deletion_file) |del_file| {
            // DeletionFile message with path field
            var del_encoder = ProtoEncoder.init(self.allocator);
            defer del_encoder.deinit();
            try del_encoder.writeStringField(1, del_file);
            try encoder.writeMessageField(3, del_encoder.getBytes());
        }

        // Field 4: physical_rows
        try encoder.writeVarintField(4, fragment.physical_rows);

        return try self.allocator.dupe(u8, encoder.getBytes());
    }

    fn encodeDataFile(self: *Self, file: DataFile) ![]const u8 {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        // Field 1: path
        try encoder.writeStringField(1, file.path);

        // Field 2: fields (repeated uint32)
        if (file.fields.len > 0) {
            for (file.fields) |field_id| {
                try encoder.writeVarintField(2, field_id);
            }
        }

        // Field 3: column_indices (repeated uint32)
        if (file.column_indices.len > 0) {
            for (file.column_indices) |col_idx| {
                try encoder.writeVarintField(3, col_idx);
            }
        }

        return try self.allocator.dupe(u8, encoder.getBytes());
    }

    fn encodeTimestamp(self: *Self, ts: Timestamp) ![]const u8 {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        // Field 1: seconds (int64 as varint)
        const seconds_u64: u64 = @bitCast(ts.seconds);
        try encoder.writeVarintField(1, seconds_u64);

        // Field 2: nanos (int32 as varint)
        if (ts.nanos != 0) {
            const nanos_u32: u32 = @bitCast(ts.nanos);
            try encoder.writeVarintField(2, nanos_u32);
        }

        return try self.allocator.dupe(u8, encoder.getBytes());
    }

    fn encodeWriterVersion(self: *Self) ![]const u8 {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        // WriterVersion message
        // Field 1: library (string)
        try encoder.writeStringField(1, WRITER_VERSION);

        return try self.allocator.dupe(u8, encoder.getBytes());
    }
};

/// Generate a UUID v4 string for transaction IDs
pub fn generateUUID(allocator: std.mem.Allocator) ![]const u8 {
    var bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&bytes);

    // Set version 4 and variant bits
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;

    return try std.fmt.allocPrint(allocator, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
        bytes[0],  bytes[1],  bytes[2],  bytes[3],
        bytes[4],  bytes[5],  bytes[6],  bytes[7],
        bytes[8],  bytes[9],  bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15],
    });
}

// ============================================================================
// Tests
// ============================================================================

test "manifest writer: basic manifest" {
    const allocator = std.testing.allocator;

    var writer = ManifestWriter.init(allocator);
    defer writer.deinit();

    // Add schema
    try writer.addField(.{
        .name = "id",
        .id = 0,
        .logical_type = "int64",
    });
    try writer.addField(.{
        .name = "value",
        .id = 1,
        .logical_type = "double",
    });

    // Add fragment
    try writer.addFragment(.{
        .id = 0,
        .files = &[_]DataFile{.{
            .path = "data/test.lance",
        }},
        .physical_rows = 100,
    });

    writer.setVersion(1);
    writer.setTransactionId("test-txn-123");

    const data = try writer.encode();
    defer allocator.free(data);

    // Verify magic at end
    try std.testing.expectEqualSlices(u8, MANIFEST_MAGIC, data[data.len - 4 ..]);

    // Verify we have data
    try std.testing.expect(data.len > 20);
}

test "manifest writer: multiple fragments" {
    const allocator = std.testing.allocator;

    var writer = ManifestWriter.init(allocator);
    defer writer.deinit();

    // Add schema
    try writer.addField(.{
        .name = "text",
        .id = 0,
        .logical_type = "string",
    });

    // Add multiple fragments
    try writer.addFragment(.{
        .id = 0,
        .files = &[_]DataFile{.{ .path = "data/frag0.lance" }},
        .physical_rows = 1000,
    });
    try writer.addFragment(.{
        .id = 1,
        .files = &[_]DataFile{.{ .path = "data/frag1.lance" }},
        .physical_rows = 2000,
    });

    writer.setVersion(2);

    const data = try writer.encode();
    defer allocator.free(data);

    // Verify magic
    try std.testing.expectEqualSlices(u8, MANIFEST_MAGIC, data[data.len - 4 ..]);

    // Verify next_row_id accumulated correctly
    try std.testing.expectEqual(@as(u64, 3000), writer.next_row_id);
}

test "generate uuid" {
    const allocator = std.testing.allocator;

    const uuid1 = try generateUUID(allocator);
    defer allocator.free(uuid1);

    const uuid2 = try generateUUID(allocator);
    defer allocator.free(uuid2);

    // UUIDs should be different
    try std.testing.expect(!std.mem.eql(u8, uuid1, uuid2));

    // UUID should have correct format (8-4-4-4-12 = 36 chars)
    try std.testing.expectEqual(@as(usize, 36), uuid1.len);

    // Check dashes in correct positions
    try std.testing.expectEqual(@as(u8, '-'), uuid1[8]);
    try std.testing.expectEqual(@as(u8, '-'), uuid1[13]);
    try std.testing.expectEqual(@as(u8, '-'), uuid1[18]);
    try std.testing.expectEqual(@as(u8, '-'), uuid1[23]);
}

test "timestamp now" {
    const ts = Timestamp.now();

    // Should be a reasonable timestamp (after 2020)
    try std.testing.expect(ts.seconds > 1577836800); // 2020-01-01

    // Nanos should be in valid range
    try std.testing.expect(ts.nanos >= 0);
    try std.testing.expect(ts.nanos < 1_000_000_000);
}

test "timestamp from millis" {
    const ts = Timestamp.fromMillis(1705315800123); // 2024-01-15 10:30:00.123 UTC

    try std.testing.expectEqual(@as(i64, 1705315800), ts.seconds);
    try std.testing.expectEqual(@as(i32, 123_000_000), ts.nanos);
}
