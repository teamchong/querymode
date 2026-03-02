//! Text and Parquet Format Ingesters
//!
//! CSV, JSON/JSONL, and Parquet ingestion to Lance format.

const std = @import("std");
const lanceql = @import("lanceql");
const csv = lanceql.encoding.csv;
const json = lanceql.encoding.json;
const writer = lanceql.encoding.writer;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const parquet_meta = lanceql.format.parquet_metadata;

/// Convert CSV column type to Lance data type
pub fn csvTypeToLanceType(csv_type: csv.ColumnType) writer.DataType {
    return switch (csv_type) {
        .int64 => .int64,
        .float64 => .float64,
        .bool_ => .bool,
        .string => .string,
    };
}

/// Finalize Lance writer and write to output file
pub fn finalizeLanceFile(lance_writer: *writer.LanceWriter, output_path: []const u8) !void {
    const lance_data = try lance_writer.finalize();
    const out_file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
        std.debug.print("Error creating output file: {}\n", .{err});
        return error.WriteError;
    };
    defer out_file.close();
    out_file.writeAll(lance_data) catch |err| {
        std.debug.print("Error writing output file: {}\n", .{err});
        return error.WriteError;
    };
    std.debug.print("Created: {s} ({d} bytes)\n", .{ output_path, lance_data.len });
}

/// Ingest CSV data to Lance file
pub fn ingestCsv(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
    config: csv.Config,
) !void {
    std.debug.print("Parsing CSV...\n", .{});

    const result = try csv.readCsv(allocator, data, config);
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    std.debug.print("  Rows: {d}\n", .{result.row_count});
    std.debug.print("  Columns: {d}\n", .{result.columns.len});

    for (result.columns) |col| {
        std.debug.print("    - {s}: {s}\n", .{ col.name, col.col_type.format() });
    }

    var schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = csvTypeToLanceType(col.col_type),
        };
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    std.debug.print("Writing Lance file...\n", .{});

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    for (result.columns, 0..) |col, i| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;

        switch (col.col_type) {
            .int64 => try encoder.writeInt64Slice(col.int64_values.items),
            .float64 => try encoder.writeFloat64Slice(col.float64_values.items),
            .bool_ => try encoder.writeBools(col.bool_values.items),
            .string => {
                try encoder.writeStrings(col.string_values.items, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = encoder.getBytes(),
            .row_count = @intCast(col.len()),
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Ingest JSON/JSONL data to Lance file
pub fn ingestJson(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
    config: json.Config,
) !void {
    std.debug.print("Parsing JSON...\n", .{});

    const detected_format = json.detectFormat(data);
    std.debug.print("  Format: {s}\n", .{detected_format.format()});

    const result = json.readJson(allocator, data, config) catch |err| {
        std.debug.print("Error parsing JSON: {}\n", .{err});
        return;
    };
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    std.debug.print("  Rows: {d}\n", .{result.row_count});
    std.debug.print("  Columns: {d}\n", .{result.columns.len});

    for (result.columns) |col| {
        std.debug.print("    - {s}: {s}\n", .{ col.name, col.col_type.format() });
    }

    var schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = csvTypeToLanceType(col.col_type),
        };
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    std.debug.print("Writing Lance file...\n", .{});

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    for (result.columns, 0..) |col, i| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;

        switch (col.col_type) {
            .int64 => try encoder.writeInt64Slice(col.int64_values.items),
            .float64 => try encoder.writeFloat64Slice(col.float64_values.items),
            .bool_ => try encoder.writeBools(col.bool_values.items),
            .string => {
                try encoder.writeStrings(col.string_values.items, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = encoder.getBytes(),
            .row_count = @intCast(col.len()),
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    try finalizeLanceFile(&lance_writer, output_path);
}

/// Map Parquet physical type to Lance DataType
fn parquetTypeToLanceType(pq_type: parquet_meta.Type) writer.DataType {
    return switch (pq_type) {
        .boolean => .bool,
        .int32 => .int32,
        .int64 => .int64,
        .float => .float32,
        .double => .float64,
        .byte_array => .string,
        .fixed_len_byte_array => .string,
        .int96 => .int64,
        _ => .string,
    };
}

/// Ingest Parquet data to Lance file
pub fn ingestParquet(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing Parquet file...\n", .{});

    var pq_table = ParquetTable.init(allocator, data) catch |err| {
        std.debug.print("Error parsing Parquet: {}\n", .{err});
        return;
    };
    defer pq_table.deinit();

    const num_cols = pq_table.numColumns();
    const num_rows = pq_table.numRows();
    const col_names = pq_table.getColumnNames();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});

    var schema = try allocator.alloc(writer.ColumnSchema, num_cols);
    defer allocator.free(schema);

    for (0..num_cols) |i| {
        const pq_type = pq_table.getColumnType(i) orelse .byte_array;
        const lance_type = parquetTypeToLanceType(pq_type);
        schema[i] = .{
            .name = col_names[i],
            .data_type = lance_type,
        };
        std.debug.print("    - {s}: {s} -> {s}\n", .{
            col_names[i],
            @tagName(pq_type),
            @tagName(lance_type),
        });
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (0..num_cols) |col_idx| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;
        var row_count: u32 = 0;

        const lance_type = schema[col_idx].data_type;

        switch (lance_type) {
            .int64 => {
                const values = pq_table.readInt64Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read int64 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);
                row_count = @intCast(values.len);
            },
            .int32 => {
                const values = pq_table.readInt32Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read int32 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt32Slice(values);
                row_count = @intCast(values.len);
            },
            .float64 => {
                const values = pq_table.readFloat64Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read float64 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);
                row_count = @intCast(values.len);
            },
            .float32 => {
                const values = pq_table.readFloat32Column(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read float32 column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat32Slice(values);
                row_count = @intCast(values.len);
            },
            .bool => {
                const values = pq_table.readBoolColumn(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read bool column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeBools(values);
                row_count = @intCast(values.len);
            },
            .string => {
                const values = pq_table.readStringColumn(col_idx) catch |err| {
                    std.debug.print("Error: Failed to read string column {d}: {}\n", .{ col_idx, err });
                    return;
                };
                defer {
                    for (values) |s| allocator.free(s);
                    allocator.free(values);
                }
                try encoder.writeStrings(values, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
                row_count = @intCast(values.len);
            },
            else => {
                std.debug.print("  Skipping unsupported type: {s}\n", .{@tagName(lance_type)});
                continue;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(col_idx),
            .data = encoder.getBytes(),
            .row_count = row_count,
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    try finalizeLanceFile(&lance_writer, output_path);
}
