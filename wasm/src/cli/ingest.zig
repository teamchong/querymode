//! LanceQL Ingest Command
//!
//! Converts data files to Lance format.
//! Supports: CSV, TSV, JSON, JSONL, Parquet, Arrow, Avro, ORC, XLSX, Delta, Iceberg
//!
//! Usage:
//!   lanceql ingest data.csv -o output.lance
//!   lanceql ingest data.json --format jsonl -o output.lance
//!   lanceql ingest data.arrow -o output.lance
//!   lanceql ingest ./delta_table/ --format delta -o output.lance

const std = @import("std");
const lanceql = @import("lanceql");
const csv = lanceql.encoding.csv;
const delta = lanceql.encoding.delta;
const iceberg = lanceql.encoding.iceberg;
const args = @import("args.zig");

// Ingest modules
const formats = @import("ingest/formats.zig");
const binary_formats = @import("ingest/binary_formats.zig");
const lakehouse = @import("ingest/lakehouse.zig");

pub const IngestError = error{
    NoInputFile,
    NoOutputFile,
    UnsupportedFormat,
    FileReadError,
    WriteError,
    OutOfMemory,
};

/// Detect file format from extension or directory structure
fn detectFormat(allocator: std.mem.Allocator, path: []const u8) args.IngestOptions.Format {
    // File extension detection
    if (std.mem.endsWith(u8, path, ".csv")) return .csv;
    if (std.mem.endsWith(u8, path, ".tsv")) return .tsv;
    if (std.mem.endsWith(u8, path, ".json")) return .json;
    if (std.mem.endsWith(u8, path, ".jsonl") or std.mem.endsWith(u8, path, ".ndjson")) return .jsonl;
    if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
    if (std.mem.endsWith(u8, path, ".arrow") or std.mem.endsWith(u8, path, ".arrows") or std.mem.endsWith(u8, path, ".feather")) return .arrow;
    if (std.mem.endsWith(u8, path, ".avro")) return .avro;
    if (std.mem.endsWith(u8, path, ".orc")) return .orc;
    if (std.mem.endsWith(u8, path, ".xlsx") or std.mem.endsWith(u8, path, ".xls")) return .xlsx;

    // Directory-based formats - check for Delta Lake (_delta_log) or Iceberg (metadata/)
    if (delta.DeltaReader.isValid(path)) return .delta;
    if (iceberg.IcebergReader.isValid(path)) return .iceberg;

    // Try magic-byte detection for files without extension
    if (detectFormatFromContent(allocator, path)) |fmt| return fmt;

    return .csv; // default to CSV
}

/// Detect format from file magic bytes
fn detectFormatFromContent(allocator: std.mem.Allocator, path: []const u8) ?args.IngestOptions.Format {
    const file = std.fs.cwd().openFile(path, .{}) catch return null;
    defer file.close();

    // Read first few bytes for magic detection
    var header: [16]u8 = undefined;
    const bytes_read = file.read(&header) catch return null;
    if (bytes_read < 4) return null;

    const data = header[0..bytes_read];

    // Arrow IPC: "ARROW1"
    if (bytes_read >= 6 and std.mem.eql(u8, data[0..6], "ARROW1")) return .arrow;

    // Avro: "Obj\x01"
    if (bytes_read >= 4 and std.mem.eql(u8, data[0..4], &[_]u8{ 'O', 'b', 'j', 1 })) return .avro;

    // ORC: "ORC" at start
    if (bytes_read >= 3 and std.mem.eql(u8, data[0..3], "ORC")) return .orc;

    // XLSX/ZIP: "PK\x03\x04"
    if (bytes_read >= 4 and std.mem.readInt(u32, data[0..4], .little) == 0x04034b50) return .xlsx;

    // Parquet: "PAR1"
    if (bytes_read >= 4 and std.mem.eql(u8, data[0..4], "PAR1")) return .parquet;

    // JSON: starts with '{' or '['
    if (data[0] == '{' or data[0] == '[') {
        // Check if it's JSONL (multiple objects separated by newlines)
        const full_data = file.readToEndAlloc(allocator, 1024) catch return .json;
        defer allocator.free(full_data);
        if (std.mem.indexOf(u8, full_data, "}\n{") != null) return .jsonl;
        return .json;
    }

    return null;
}

/// Run the ingest command
pub fn run(allocator: std.mem.Allocator, opts: args.IngestOptions) !void {
    const input_path = opts.input orelse {
        std.debug.print("Error: Input file required.\n", .{});
        args.printIngestHelp();
        return;
    };

    const output_path = opts.output orelse {
        std.debug.print("Error: Output file required. Use -o <output.lance>\n", .{});
        return;
    };

    // Determine format
    const format = if (opts.format == .auto) detectFormat(allocator, input_path) else opts.format;
    std.debug.print("Format: {s}\n", .{@tagName(format)});

    // Directory-based formats don't need file reading
    switch (format) {
        .delta => {
            try lakehouse.ingestDelta(allocator, input_path, output_path);
            return;
        },
        .iceberg => {
            try lakehouse.ingestIceberg(allocator, input_path, output_path);
            return;
        },
        else => {},
    }

    // Read input file
    const file = std.fs.cwd().openFile(input_path, .{}) catch |err| {
        std.debug.print("Error opening '{s}': {}\n", .{ input_path, err });
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch |err| {
        std.debug.print("Error reading file: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    // Process based on format
    switch (format) {
        .csv, .tsv => {
            try formats.ingestCsv(allocator, data, output_path, .{
                .delimiter = if (format == .tsv) '\t' else opts.delimiter,
                .has_header = opts.header,
            });
        },
        .json, .jsonl => {
            try formats.ingestJson(allocator, data, output_path, .{});
        },
        .parquet => {
            try formats.ingestParquet(allocator, data, output_path);
        },
        .arrow => {
            try binary_formats.ingestArrow(allocator, data, output_path);
        },
        .avro => {
            try binary_formats.ingestAvro(allocator, data, output_path);
        },
        .orc => {
            try binary_formats.ingestOrc(allocator, data, output_path);
        },
        .xlsx => {
            try binary_formats.ingestXlsx(allocator, data, output_path);
        },
        .delta, .iceberg => unreachable, // Handled above
        .auto => unreachable,
    }
}

test "detect format by extension" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(args.IngestOptions.Format.csv, detectFormat(allocator, "data.csv"));
    try std.testing.expectEqual(args.IngestOptions.Format.tsv, detectFormat(allocator, "data.tsv"));
    try std.testing.expectEqual(args.IngestOptions.Format.json, detectFormat(allocator, "data.json"));
    try std.testing.expectEqual(args.IngestOptions.Format.jsonl, detectFormat(allocator, "data.jsonl"));
    try std.testing.expectEqual(args.IngestOptions.Format.parquet, detectFormat(allocator, "data.parquet"));
    try std.testing.expectEqual(args.IngestOptions.Format.arrow, detectFormat(allocator, "data.arrow"));
    try std.testing.expectEqual(args.IngestOptions.Format.arrow, detectFormat(allocator, "data.feather"));
    try std.testing.expectEqual(args.IngestOptions.Format.avro, detectFormat(allocator, "data.avro"));
    try std.testing.expectEqual(args.IngestOptions.Format.orc, detectFormat(allocator, "data.orc"));
    try std.testing.expectEqual(args.IngestOptions.Format.xlsx, detectFormat(allocator, "data.xlsx"));
}

test "detect format from fixtures" {
    const allocator = std.testing.allocator;

    // Delta Lake directory
    try std.testing.expectEqual(args.IngestOptions.Format.delta, detectFormat(allocator, "tests/fixtures/simple.delta"));

    // Iceberg directory
    try std.testing.expectEqual(args.IngestOptions.Format.iceberg, detectFormat(allocator, "tests/fixtures/simple.iceberg"));
}
