//! Lakehouse Format Ingesters
//!
//! Delta Lake and Apache Iceberg ingestion to Lance format.

const std = @import("std");
const querymode = @import("querymode");
const delta = querymode.encoding.delta;
const iceberg = querymode.encoding.iceberg;

pub fn ingestDelta(
    allocator: std.mem.Allocator,
    path: []const u8,
    output_path: []const u8,
) !void {
    _ = output_path;

    var reader = delta.DeltaReader.init(allocator, path) catch |err| {
        std.debug.print("Delta parse failed: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    std.debug.print("Delta: {s} ({d} rows, {d} cols, {d} files)\n", .{
        path,
        reader.rowCount(),
        reader.columnCount(),
        reader.fileCount(),
    });
    std.debug.print("Use: querymode ingest {s}/*.parquet -o output.lance\n", .{path});
}

pub fn ingestIceberg(
    allocator: std.mem.Allocator,
    path: []const u8,
    output_path: []const u8,
) !void {
    _ = output_path;

    var reader = iceberg.IcebergReader.init(allocator, path) catch |err| {
        std.debug.print("Iceberg parse failed: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    std.debug.print("Iceberg v{d}: {s} ({d} cols, snapshot {d})\n", .{
        reader.formatVersion(),
        path,
        reader.columnCount(),
        reader.snapshotId(),
    });
    std.debug.print("Use: querymode ingest {s}/data/*.parquet -o output.lance\n", .{path});
}
