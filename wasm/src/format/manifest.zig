//! Lance manifest file parser.
//!
//! Manifests are stored in `_versions/N.manifest` files within a Lance dataset.
//! They contain:
//! - Schema (field definitions)
//! - Fragments (data file references)
//! - Version metadata (number, timestamp, transaction ID)
//!
//! ## File Layout
//! | Offset | Size | Content |
//! |--------|------|---------|
//! | 0 | 4 | Protobuf length (little-endian u32) |
//! | 4 | N | Protobuf-encoded Manifest message |
//! | 4+N | 12 | Footer (padding + metadata) |
//! | EOF-4 | 4 | Magic "LANC" |
//!
//! ## Manifest Protobuf Fields (from table.proto)
//! | Field | Number | Type |
//! |-------|--------|------|
//! | fields | 1 | repeated Field (schema) |
//! | fragments | 2 | repeated DataFragment |
//! | version | 3 | uint64 |
//! | timestamp | 7 | google.protobuf.Timestamp |
//! | reader_feature_flags | 9 | uint64 |
//! | writer_feature_flags | 10 | uint64 |
//! | next_row_id | 14 | uint64 |
//! | writer_version | 13 | WriterVersion |

const std = @import("std");
const proto_mod = @import("lanceql.proto");

const ProtoDecoder = proto_mod.ProtoDecoder;
const DecodeError = proto_mod.DecodeError;

/// Magic bytes identifying a Lance manifest
pub const MANIFEST_MAGIC = "LANC";

/// Minimum size of a manifest file (4-byte header + empty proto + footer)
pub const MIN_MANIFEST_SIZE: usize = 20;

/// Error types for manifest parsing
pub const ManifestError = error{
    /// Invalid magic bytes
    InvalidMagic,
    /// File too small to be a valid manifest
    FileTooSmall,
    /// Protobuf decoding error
    DecodeError,
    /// Memory allocation failed
    OutOfMemory,
};

/// A single data fragment in the dataset
pub const Fragment = struct {
    /// Fragment ID (unique within version)
    id: u64,
    /// Relative path to the data file
    file_path: []const u8,
    /// Number of physical rows in this fragment
    physical_rows: u64,
    /// Whether this fragment has a deletion file
    has_deletion: bool,

    pub fn deinit(self: *Fragment, allocator: std.mem.Allocator) void {
        allocator.free(self.file_path);
    }
};

/// Google protobuf Timestamp (seconds since epoch + nanoseconds)
pub const Timestamp = struct {
    seconds: i64,
    nanos: i32,

    /// Convert to milliseconds since Unix epoch
    pub fn toMillis(self: Timestamp) i64 {
        return self.seconds * 1000 + @divTrunc(self.nanos, 1_000_000);
    }

    /// Format as ISO 8601 string (YYYY-MM-DDTHH:MM:SS.sssZ)
    pub fn format(self: Timestamp, allocator: std.mem.Allocator) ![]const u8 {
        const epoch_seconds: u64 = @intCast(self.seconds);
        const epoch_day = epoch_seconds / 86400;
        const day_seconds = epoch_seconds % 86400;

        // Calculate date (simplified, doesn't handle pre-1970)
        var year: u32 = 1970;
        var remaining_days: u64 = epoch_day;

        while (true) {
            const days_in_year: u64 = if (isLeapYear(year)) 366 else 365;
            if (remaining_days < days_in_year) break;
            remaining_days -= days_in_year;
            year += 1;
        }

        // Month calculation
        const month_days = if (isLeapYear(year))
            [_]u8{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
        else
            [_]u8{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

        var month: u32 = 1;
        for (month_days) |days| {
            if (remaining_days < days) break;
            remaining_days -= days;
            month += 1;
        }
        const day: u32 = @intCast(remaining_days + 1);

        // Time calculation
        const hours: u32 = @intCast(day_seconds / 3600);
        const minutes: u32 = @intCast((day_seconds % 3600) / 60);
        const secs: u32 = @intCast(day_seconds % 60);
        const millis: u32 = @intCast(@divTrunc(@as(u32, @intCast(self.nanos)), 1_000_000));

        return std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}Z", .{
            year,
            month,
            day,
            hours,
            minutes,
            secs,
            millis,
        });
    }

    fn isLeapYear(year: u32) bool {
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0);
    }
};

/// Parsed manifest from a Lance dataset version
pub const Manifest = struct {
    /// Version number (1-based)
    version: u64,
    /// Timestamp when this version was created
    timestamp: Timestamp,
    /// All fragments in this version
    fragments: []Fragment,
    /// Transaction ID (UUID string)
    transaction_id: []const u8,
    /// Total physical rows across all fragments
    total_rows: u64,
    /// Allocator used for memory
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Parse a manifest from raw file bytes
    pub fn parse(allocator: std.mem.Allocator, data: []const u8) ManifestError!Self {
        if (data.len < MIN_MANIFEST_SIZE) {
            return ManifestError.FileTooSmall;
        }

        // Verify magic at end
        if (!std.mem.eql(u8, data[data.len - 4 ..], MANIFEST_MAGIC)) {
            return ManifestError.InvalidMagic;
        }

        // Read protobuf length (first 4 bytes, little-endian)
        const proto_len = std.mem.readInt(u32, data[0..4], .little);
        if (4 + proto_len > data.len) {
            return ManifestError.DecodeError;
        }

        // Parse protobuf content
        const proto_data = data[4 .. 4 + proto_len];
        return parseProtobuf(allocator, proto_data);
    }

    fn parseProtobuf(allocator: std.mem.Allocator, data: []const u8) ManifestError!Self {
        var proto = ProtoDecoder.init(data);

        var version: u64 = 0;
        var timestamp = Timestamp{ .seconds = 0, .nanos = 0 };
        var fragments = std.ArrayListUnmanaged(Fragment){};
        var transaction_id: ?[]const u8 = null;
        var total_rows: u64 = 0;

        errdefer {
            for (fragments.items) |*f| {
                f.deinit(allocator);
            }
            fragments.deinit(allocator);
            if (transaction_id) |tid| allocator.free(tid);
        }

        while (proto.hasMore()) {
            const header = proto.readFieldHeader() catch return ManifestError.DecodeError;

            switch (header.field_num) {
                1 => {
                    // Schema fields - skip for now (we read schema from data files)
                    proto.skipField(header.wire_type) catch return ManifestError.DecodeError;
                },
                2 => {
                    // DataFragment
                    const frag_bytes = proto.readBytes() catch return ManifestError.DecodeError;
                    const frag = parseFragment(allocator, frag_bytes) catch return ManifestError.DecodeError;
                    total_rows += frag.physical_rows;
                    fragments.append(allocator, frag) catch return ManifestError.OutOfMemory;
                },
                3 => {
                    // version (uint64)
                    version = proto.readVarint() catch return ManifestError.DecodeError;
                },
                7 => {
                    // timestamp (google.protobuf.Timestamp)
                    const ts_bytes = proto.readBytes() catch return ManifestError.DecodeError;
                    timestamp = parseTimestamp(ts_bytes) catch return ManifestError.DecodeError;
                },
                12 => {
                    // Transaction ID (string) - field 12 based on hex analysis
                    const tid_bytes = proto.readBytes() catch return ManifestError.DecodeError;
                    if (transaction_id) |old| allocator.free(old);
                    transaction_id = allocator.dupe(u8, tid_bytes) catch return ManifestError.OutOfMemory;
                },
                else => {
                    proto.skipField(header.wire_type) catch return ManifestError.DecodeError;
                },
            }
        }

        return Self{
            .version = version,
            .timestamp = timestamp,
            .fragments = fragments.toOwnedSlice(allocator) catch return ManifestError.OutOfMemory,
            .transaction_id = transaction_id orelse (allocator.alloc(u8, 0) catch return ManifestError.OutOfMemory),
            .total_rows = total_rows,
            .allocator = allocator,
        };
    }

    fn parseFragment(allocator: std.mem.Allocator, data: []const u8) DecodeError!Fragment {
        var proto = ProtoDecoder.init(data);

        var id: u64 = 0;
        var file_path: ?[]const u8 = null;
        var physical_rows: u64 = 0;
        var has_deletion: bool = false;

        errdefer if (file_path) |fp| allocator.free(fp);

        while (proto.hasMore()) {
            const header = try proto.readFieldHeader();

            switch (header.field_num) {
                1 => {
                    // id (uint64) - but may be packed in files field
                    if (header.wire_type == .varint) {
                        id = try proto.readVarint();
                    } else {
                        try proto.skipField(header.wire_type);
                    }
                },
                2 => {
                    // files (repeated DataFile) - extract path from first file
                    const file_bytes = try proto.readBytes();
                    if (file_path == null) {
                        file_path = try parseDataFilePath(allocator, file_bytes);
                    }
                },
                3 => {
                    // deletion_file
                    has_deletion = true;
                    try proto.skipField(header.wire_type);
                },
                4 => {
                    // physical_rows (uint64)
                    physical_rows = try proto.readVarint();
                },
                else => {
                    try proto.skipField(header.wire_type);
                },
            }
        }

        return Fragment{
            .id = id,
            .file_path = file_path orelse (allocator.alloc(u8, 0) catch return DecodeError.OutOfMemory),
            .physical_rows = physical_rows,
            .has_deletion = has_deletion,
        };
    }

    fn parseDataFilePath(allocator: std.mem.Allocator, data: []const u8) DecodeError![]const u8 {
        var proto = ProtoDecoder.init(data);

        while (proto.hasMore()) {
            const header = try proto.readFieldHeader();

            switch (header.field_num) {
                1 => {
                    // path (string)
                    const path_bytes = try proto.readBytes();
                    return allocator.dupe(u8, path_bytes) catch return DecodeError.OutOfMemory;
                },
                else => {
                    try proto.skipField(header.wire_type);
                },
            }
        }

        return allocator.alloc(u8, 0) catch return DecodeError.OutOfMemory;
    }

    fn parseTimestamp(data: []const u8) DecodeError!Timestamp {
        var proto = ProtoDecoder.init(data);
        var seconds: i64 = 0;
        var nanos: i32 = 0;

        while (proto.hasMore()) {
            const header = try proto.readFieldHeader();

            switch (header.field_num) {
                1 => {
                    // seconds (int64)
                    const val = try proto.readVarint();
                    seconds = @bitCast(val);
                },
                2 => {
                    // nanos (int32)
                    const val = try proto.readVarint();
                    nanos = @intCast(@as(u32, @truncate(val)));
                },
                else => {
                    try proto.skipField(header.wire_type);
                },
            }
        }

        return Timestamp{ .seconds = seconds, .nanos = nanos };
    }

    pub fn deinit(self: *Self) void {
        for (self.fragments) |*f| {
            var frag = f.*;
            frag.deinit(self.allocator);
        }
        self.allocator.free(self.fragments);
        self.allocator.free(self.transaction_id);
    }

    /// Get fragment count
    pub fn fragmentCount(self: Self) usize {
        return self.fragments.len;
    }
};

/// List all manifest versions in a dataset directory
pub fn listVersions(allocator: std.mem.Allocator, dataset_path: []const u8) ![]u64 {
    const versions_path = try std.fs.path.join(allocator, &.{ dataset_path, "_versions" });
    defer allocator.free(versions_path);

    var dir = std.fs.cwd().openDir(versions_path, .{ .iterate = true }) catch |err| {
        if (err == error.FileNotFound) return &[_]u64{};
        return err;
    };
    defer dir.close();

    var versions = std.ArrayListUnmanaged(u64){};
    errdefer versions.deinit(allocator);

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".manifest")) continue;

        // Parse version number from filename (e.g., "1.manifest" -> 1)
        const base = entry.name[0 .. entry.name.len - 9]; // Remove ".manifest"
        const version = std.fmt.parseInt(u64, base, 10) catch continue;
        try versions.append(allocator, version);
    }

    // Sort versions
    const items = try versions.toOwnedSlice(allocator);
    std.mem.sort(u64, items, {}, std.sort.asc(u64));
    return items;
}

/// Load a specific manifest version
pub fn loadManifest(allocator: std.mem.Allocator, dataset_path: []const u8, version: u64) !Manifest {
    var buf: [32]u8 = undefined;
    const version_str = std.fmt.bufPrint(&buf, "{d}.manifest", .{version}) catch unreachable;

    const manifest_path = try std.fs.path.join(allocator, &.{ dataset_path, "_versions", version_str });
    defer allocator.free(manifest_path);

    const file = try std.fs.cwd().openFile(manifest_path, .{});
    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);

    const bytes_read = try file.readAll(data);
    if (bytes_read != stat.size) {
        return error.UnexpectedEof;
    }

    return Manifest.parse(allocator, data);
}

/// Get the latest version number in a dataset
pub fn latestVersion(allocator: std.mem.Allocator, dataset_path: []const u8) !?u64 {
    const versions = try listVersions(allocator, dataset_path);
    defer allocator.free(versions);

    if (versions.len == 0) return null;
    return versions[versions.len - 1];
}

// ============================================================================
// Tests
// ============================================================================

test "parse manifest from multiple_batches fixture" {
    const allocator = std.testing.allocator;

    // Load v2 manifest (has 2 fragments)
    var manifest = try loadManifest(allocator, "tests/fixtures/multiple_batches.lance", 2);
    defer manifest.deinit();

    std.debug.print("\nManifest v2:\n", .{});
    std.debug.print("  version: {d}\n", .{manifest.version});
    std.debug.print("  fragments: {d}\n", .{manifest.fragments.len});
    std.debug.print("  total_rows: {d}\n", .{manifest.total_rows});
    std.debug.print("  timestamp: {d}s\n", .{manifest.timestamp.seconds});
    std.debug.print("  transaction_id: {s}\n", .{manifest.transaction_id});

    for (manifest.fragments, 0..) |frag, i| {
        std.debug.print("  fragment[{d}]: id={d}, rows={d}, path={s}\n", .{
            i,
            frag.id,
            frag.physical_rows,
            frag.file_path,
        });
    }

    // Verify
    try std.testing.expectEqual(@as(u64, 2), manifest.version);
    try std.testing.expectEqual(@as(usize, 2), manifest.fragments.len);
}

test "list versions in multiple_batches fixture" {
    const allocator = std.testing.allocator;

    const versions = try listVersions(allocator, "tests/fixtures/multiple_batches.lance");
    defer allocator.free(versions);

    std.debug.print("\nVersions found: ", .{});
    for (versions) |v| {
        std.debug.print("{d} ", .{v});
    }
    std.debug.print("\n", .{});

    try std.testing.expectEqual(@as(usize, 3), versions.len);
    try std.testing.expectEqual(@as(u64, 1), versions[0]);
    try std.testing.expectEqual(@as(u64, 2), versions[1]);
    try std.testing.expectEqual(@as(u64, 3), versions[2]);
}

test "latest version" {
    const allocator = std.testing.allocator;

    const latest = try latestVersion(allocator, "tests/fixtures/multiple_batches.lance");
    try std.testing.expectEqual(@as(?u64, 3), latest);
}

test "timestamp formatting" {
    const allocator = std.testing.allocator;

    // 2024-01-15 10:30:00 UTC
    const ts = Timestamp{ .seconds = 1705315800, .nanos = 123_000_000 };
    const formatted = try ts.format(allocator);
    defer allocator.free(formatted);

    std.debug.print("\nFormatted timestamp: {s}\n", .{formatted});
    try std.testing.expectEqualStrings("2024-01-15T10:30:00.123Z", formatted);
}
