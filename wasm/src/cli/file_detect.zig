//! Unified File Type Detection
//!
//! Consolidates file format detection logic used across CLI commands.
//! Detects Lance, Parquet, Arrow, Avro, ORC, XLSX, Delta, and Iceberg formats.

const std = @import("std");

/// Supported file formats
pub const FileType = enum {
    lance,
    parquet,
    delta,
    iceberg,
    arrow,
    avro,
    orc,
    xlsx,
    csv,
    json,
    jsonl,
    tsv,
    unknown,
};

/// Magic byte signatures for binary formats
pub const Magic = struct {
    pub const ARROW: *const [6]u8 = "ARROW1";
    pub const PARQUET: *const [4]u8 = "PAR1";
    pub const AVRO: *const [4]u8 = "Obj\x01";
    pub const ZIP: *const [4]u8 = "PK\x03\x04"; // XLSX
    pub const ORC: *const [3]u8 = "ORC";
    pub const LANCE: *const [4]u8 = "LANC";
};

/// Detect file type from path and data
pub fn detect(path: []const u8, data: []const u8) FileType {
    // Check extension first
    if (detectFromPath(path)) |ft| return ft;

    // Fall back to magic bytes
    return detectFromData(data);
}

/// Detect file type from path/extension only
pub fn detectFromPath(path: []const u8) ?FileType {
    // Binary columnar formats
    if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
    if (std.mem.endsWith(u8, path, ".lance")) return .lance;
    if (std.mem.endsWith(u8, path, ".arrow") or
        std.mem.endsWith(u8, path, ".arrows") or
        std.mem.endsWith(u8, path, ".feather")) return .arrow;
    if (std.mem.endsWith(u8, path, ".avro")) return .avro;
    if (std.mem.endsWith(u8, path, ".orc")) return .orc;
    if (std.mem.endsWith(u8, path, ".xlsx")) return .xlsx;

    // Lakehouse formats (directory-based)
    if (std.mem.endsWith(u8, path, ".delta")) return .delta;
    if (std.mem.endsWith(u8, path, ".iceberg")) return .iceberg;

    // Text formats
    if (std.mem.endsWith(u8, path, ".csv")) return .csv;
    if (std.mem.endsWith(u8, path, ".tsv")) return .tsv;
    if (std.mem.endsWith(u8, path, ".json")) return .json;
    if (std.mem.endsWith(u8, path, ".jsonl") or std.mem.endsWith(u8, path, ".ndjson")) return .jsonl;

    return null;
}

/// Detect file type from magic bytes in data
pub fn detectFromData(data: []const u8) FileType {
    if (data.len >= 6) {
        if (std.mem.eql(u8, data[0..6], Magic.ARROW)) return .arrow;
    }
    if (data.len >= 4) {
        if (std.mem.eql(u8, data[0..4], Magic.PARQUET)) return .parquet;
        if (std.mem.eql(u8, data[0..4], Magic.AVRO)) return .avro;
        if (std.mem.eql(u8, data[0..4], Magic.ZIP)) return .xlsx;
        // Lance magic at end of file
        if (data.len >= 40 and std.mem.eql(u8, data[data.len - 4 ..], Magic.LANCE)) return .lance;
    }
    if (data.len >= 3) {
        if (std.mem.eql(u8, data[0..3], Magic.ORC)) return .orc;
    }

    return .unknown;
}

/// Check if path is a Delta Lake table (directory with _delta_log/)
pub fn isDeltaDirectory(path: []const u8) bool {
    var path_buf: [4096]u8 = undefined;
    const delta_log_path = std.fmt.bufPrint(&path_buf, "{s}/_delta_log", .{path}) catch return false;
    const stat = std.fs.cwd().statFile(delta_log_path) catch return false;
    return stat.kind == .directory;
}

/// Check if path is an Iceberg table (directory with metadata/)
pub fn isIcebergDirectory(path: []const u8) bool {
    var path_buf: [4096]u8 = undefined;
    const metadata_path = std.fmt.bufPrint(&path_buf, "{s}/metadata", .{path}) catch return false;
    const stat = std.fs.cwd().statFile(metadata_path) catch return false;
    return stat.kind == .directory;
}

/// Full detection including directory checks for lakehouse formats
pub fn detectWithDirectoryCheck(path: []const u8, data: []const u8) FileType {
    // Check extension first
    if (detectFromPath(path)) |ft| return ft;

    // Check for lakehouse directories
    if (isDeltaDirectory(path)) return .delta;
    if (isIcebergDirectory(path)) return .iceberg;

    // Fall back to magic bytes
    return detectFromData(data);
}

// =============================================================================
// Tests
// =============================================================================

test "detect from extension" {
    try std.testing.expectEqual(FileType.parquet, detectFromPath("data.parquet").?);
    try std.testing.expectEqual(FileType.lance, detectFromPath("data.lance").?);
    try std.testing.expectEqual(FileType.arrow, detectFromPath("data.arrow").?);
    try std.testing.expectEqual(FileType.arrow, detectFromPath("data.feather").?);
    try std.testing.expectEqual(FileType.avro, detectFromPath("data.avro").?);
    try std.testing.expectEqual(FileType.orc, detectFromPath("data.orc").?);
    try std.testing.expectEqual(FileType.xlsx, detectFromPath("data.xlsx").?);
    try std.testing.expectEqual(FileType.csv, detectFromPath("data.csv").?);
    try std.testing.expectEqual(FileType.json, detectFromPath("data.json").?);
    try std.testing.expect(detectFromPath("unknown.bin") == null);
}

test "detect from magic bytes" {
    try std.testing.expectEqual(FileType.parquet, detectFromData("PAR1...."));
    try std.testing.expectEqual(FileType.arrow, detectFromData("ARROW1.."));
    try std.testing.expectEqual(FileType.avro, detectFromData("Obj\x01..."));
    try std.testing.expectEqual(FileType.orc, detectFromData("ORC...."));
    try std.testing.expectEqual(FileType.unknown, detectFromData("unknown"));
}
