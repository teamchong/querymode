//! Binary Format Ingesters
//!
//! Arrow IPC, Avro, ORC, and XLSX ingestion to Lance format.

const std = @import("std");
const lanceql = @import("lanceql");
const arrow_ipc = lanceql.encoding.arrow_ipc;
const avro = lanceql.encoding.avro;
const orc = lanceql.encoding.orc;
const xlsx = lanceql.encoding.xlsx;
const writer = lanceql.encoding.writer;
const formats = @import("formats.zig");

/// Map Arrow type to Lance DataType
fn arrowTypeToLanceType(arrow_type: arrow_ipc.ArrowType) writer.DataType {
    return switch (arrow_type) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => .int64,
        .float32 => .float32,
        .float64 => .float64,
        .utf8, .large_utf8, .binary, .large_binary => .string,
        .bool_type => .bool,
        else => .string,
    };
}

/// Ingest Arrow IPC data to Lance file
pub fn ingestArrow(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing Arrow IPC file...\n", .{});

    var reader = arrow_ipc.ArrowIpcReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing Arrow: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});

    if (num_cols == 0 or num_rows == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    var supported_cols = std.ArrayListUnmanaged(usize){};
    defer supported_cols.deinit(allocator);

    for (0..num_cols) |i| {
        const arrow_type = reader.getColumnType(i);
        const lance_type = arrowTypeToLanceType(arrow_type);
        switch (lance_type) {
            .int64, .float64, .string => {
                try supported_cols.append(allocator, i);
                std.debug.print("    - {s}: {s} -> {s}\n", .{
                    reader.getColumnName(i),
                    @tagName(arrow_type),
                    @tagName(lance_type),
                });
            },
            else => {
                std.debug.print("    - {s}: {s} (skipped)\n", .{
                    reader.getColumnName(i),
                    @tagName(arrow_type),
                });
            },
        }
    }

    if (supported_cols.items.len == 0) {
        std.debug.print("  No supported columns to convert\n", .{});
        return;
    }

    var schema = try allocator.alloc(writer.ColumnSchema, supported_cols.items.len);
    defer allocator.free(schema);

    for (supported_cols.items, 0..) |orig_idx, i| {
        const arrow_type = reader.getColumnType(orig_idx);
        schema[i] = .{
            .name = reader.getColumnName(orig_idx),
            .data_type = arrowTypeToLanceType(arrow_type),
        };
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    std.debug.print("Converting columns...\n", .{});

    for (supported_cols.items, 0..) |orig_idx, schema_idx| {
        encoder.reset();

        const arrow_type = reader.getColumnType(orig_idx);
        const lance_type = arrowTypeToLanceType(arrow_type);

        switch (lance_type) {
            .int64 => {
                const values = reader.readInt64Column(orig_idx) catch |err| {
                    std.debug.print("  Error reading int64 column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .float64 => {
                const values = reader.readFloat64Column(orig_idx) catch |err| {
                    std.debug.print("  Error reading float64 column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string => {
                const values = reader.readStringColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading string column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer {
                    for (values) |v| allocator.free(v);
                    allocator.free(values);
                }

                var offsets_buf = std.ArrayListUnmanaged(u8){};
                defer offsets_buf.deinit(allocator);
                try encoder.writeStrings(values, &offsets_buf, allocator);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try formats.finalizeLanceFile(&lance_writer, output_path);
}

/// Convert Avro type to Lance type
fn avroTypeToLanceType(avro_type: avro.AvroType) writer.DataType {
    return switch (avro_type) {
        .long_type, .int_type => .int64,
        .double_type, .float_type => .float64,
        .string, .bytes => .string,
        .boolean => .bool,
        else => .string,
    };
}

/// Ingest Avro data to Lance file
pub fn ingestAvro(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing Avro file...\n", .{});

    var reader = avro.AvroReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing Avro: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Codec: {s}\n", .{@tagName(reader.codec)});

    if (num_cols == 0 or num_rows == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    if (reader.codec != .null) {
        std.debug.print("  Compressed Avro ({s}) not yet supported\n", .{@tagName(reader.codec)});
        return;
    }

    var supported_cols = std.ArrayListUnmanaged(usize){};
    defer supported_cols.deinit(allocator);

    for (0..num_cols) |i| {
        const avro_type = reader.getFieldType(i) orelse continue;
        const lance_type = avroTypeToLanceType(avro_type);
        const name = reader.getFieldName(i) orelse "unknown";

        switch (avro_type) {
            .long_type, .int_type, .double_type, .float_type, .string => {
                try supported_cols.append(allocator, i);
                std.debug.print("    - {s}: {s} -> {s}\n", .{ name, @tagName(avro_type), @tagName(lance_type) });
            },
            else => {
                std.debug.print("    - {s}: {s} (skipped)\n", .{ name, @tagName(avro_type) });
            },
        }
    }

    if (supported_cols.items.len == 0) {
        std.debug.print("  No supported columns to convert\n", .{});
        return;
    }

    var schema = try allocator.alloc(writer.ColumnSchema, supported_cols.items.len);
    defer allocator.free(schema);

    for (supported_cols.items, 0..) |orig_idx, i| {
        const avro_type = reader.getFieldType(orig_idx) orelse {
            std.debug.print("Error: Missing type for column {d}\n", .{orig_idx});
            return;
        };
        schema[i] = .{
            .name = reader.getFieldName(orig_idx) orelse "unknown",
            .data_type = avroTypeToLanceType(avro_type),
        };
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (supported_cols.items, 0..) |orig_idx, schema_idx| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        const avro_type = reader.getFieldType(orig_idx) orelse continue;

        switch (avro_type) {
            .long_type, .int_type => {
                const values = reader.readLongColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading long column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .double_type, .float_type => {
                const values = reader.readDoubleColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading double column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string => {
                const values = reader.readStringColumn(orig_idx) catch |err| {
                    std.debug.print("  Error reading string column {d}: {}\n", .{orig_idx, err});
                    return;
                };
                defer {
                    for (values) |s| allocator.free(s);
                    allocator.free(values);
                }

                try encoder.writeStrings(values, &offsets_buf, allocator);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try formats.finalizeLanceFile(&lance_writer, output_path);
}

/// Convert ORC type to Lance type
fn orcTypeToLanceType(orc_type: orc.OrcType) writer.DataType {
    return switch (orc_type) {
        .long, .int, .short, .byte => .int64,
        .double, .float => .float64,
        .string, .varchar, .char, .binary => .string,
        else => .string,
    };
}

/// Ingest ORC data to Lance file
pub fn ingestOrc(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing ORC file...\n", .{});

    var reader = orc.OrcReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing ORC: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const num_cols = reader.columnCount();
    const num_rows = reader.rowCount();
    const num_stripes = reader.stripeCount();

    std.debug.print("  Rows: {d}\n", .{num_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});
    std.debug.print("  Stripes: {d}\n", .{num_stripes});
    std.debug.print("  Compression: {s}\n", .{@tagName(reader.compression)});

    if (num_cols == 0 or num_rows == 0 or num_stripes == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    const column_types = reader.column_types orelse &[_]orc.OrcType{};

    var supported_cols = std.ArrayListUnmanaged(usize){};
    defer supported_cols.deinit(allocator);

    for (0..num_cols) |i| {
        const orc_type: orc.OrcType = if (i < column_types.len) column_types[i] else .unknown;
        const lance_type = orcTypeToLanceType(orc_type);

        switch (orc_type) {
            .long, .int, .short, .byte, .double, .float, .string, .varchar, .char => {
                try supported_cols.append(allocator, i);
                std.debug.print("    - col{d}: {s} -> {s}\n", .{ i, @tagName(orc_type), @tagName(lance_type) });
            },
            .struct_type => {
                std.debug.print("    - col{d}: struct (skipped - container)\n", .{i});
            },
            else => {
                std.debug.print("    - col{d}: {s} (skipped)\n", .{ i, @tagName(orc_type) });
            },
        }
    }

    if (supported_cols.items.len == 0) {
        std.debug.print("  No supported columns to convert\n", .{});
        return;
    }

    var schema = try allocator.alloc(writer.ColumnSchema, supported_cols.items.len);
    defer allocator.free(schema);

    const column_names = reader.column_names orelse &[_][]const u8{};

    for (supported_cols.items, 0..) |orig_idx, i| {
        const orc_type: orc.OrcType = if (orig_idx < column_types.len) column_types[orig_idx] else .unknown;
        const col_name = if (orig_idx < column_names.len) column_names[orig_idx] else "";

        var name_buf: [32]u8 = undefined;
        const name = if (col_name.len > 0) col_name else blk: {
            const len = std.fmt.bufPrint(&name_buf, "col{d}", .{orig_idx}) catch "unknown";
            break :blk len;
        };

        schema[i] = .{
            .name = name,
            .data_type = orcTypeToLanceType(orc_type),
        };
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (supported_cols.items, 0..) |orig_idx, schema_idx| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        const orc_type = if (orig_idx < column_types.len) column_types[orig_idx] else .unknown;

        switch (orc_type) {
            .long, .int, .short, .byte => {
                const values = reader.readLongColumn(@intCast(orig_idx)) catch |err| {
                    std.debug.print("  Error reading long column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeInt64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .double, .float => {
                const values = reader.readDoubleColumn(@intCast(orig_idx)) catch |err| {
                    std.debug.print("  Error reading double column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer allocator.free(values);
                try encoder.writeFloat64Slice(values);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string, .varchar, .char => {
                const values = reader.readStringColumn(@intCast(orig_idx)) catch |err| {
                    std.debug.print("  Error reading string column {d}: {}\n", .{ orig_idx, err });
                    return;
                };
                defer {
                    for (values) |s| allocator.free(s);
                    allocator.free(values);
                }

                try encoder.writeStrings(values, &offsets_buf, allocator);

                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(schema_idx),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try formats.finalizeLanceFile(&lance_writer, output_path);
}

/// Infer Lance type from XLSX cell values
fn inferXlsxColumnType(reader: *const xlsx.XlsxReader, col: usize) writer.DataType {
    var has_number = false;
    var has_string = false;

    const start_row: usize = 1;
    const end_row = @min(start_row + 10, reader.rowCount());

    for (start_row..end_row) |row| {
        if (reader.getCell(row, col)) |cell| {
            switch (cell) {
                .number => has_number = true,
                .inline_string, .string, .shared_string => has_string = true,
                else => {},
            }
        }
    }

    if (has_string) return .string;
    if (has_number) return .float64;
    return .string;
}

/// Get cell value as string
fn xlsxCellToString(cell: xlsx.CellValue) ?[]const u8 {
    return switch (cell) {
        .inline_string => |s| s,
        .string => |s| s,
        else => null,
    };
}

/// Ingest XLSX data to Lance file
pub fn ingestXlsx(
    allocator: std.mem.Allocator,
    data: []const u8,
    output_path: []const u8,
) !void {
    std.debug.print("Parsing XLSX file...\n", .{});

    var reader = xlsx.XlsxReader.init(allocator, data) catch |err| {
        std.debug.print("Error parsing XLSX: {}\n", .{err});
        return;
    };
    defer reader.deinit();

    const total_rows = reader.rowCount();
    const num_cols = reader.columnCount();

    if (total_rows == 0 or num_cols == 0) {
        std.debug.print("  No data to convert\n", .{});
        return;
    }

    const num_data_rows = total_rows - 1;
    std.debug.print("  Rows: {d} (including header)\n", .{total_rows});
    std.debug.print("  Columns: {d}\n", .{num_cols});

    if (num_data_rows == 0) {
        std.debug.print("  No data rows (header only)\n", .{});
        return;
    }

    var schema = try allocator.alloc(writer.ColumnSchema, num_cols);
    defer allocator.free(schema);

    var col_names = try allocator.alloc([]const u8, num_cols);
    defer {
        for (col_names) |name| allocator.free(name);
        allocator.free(col_names);
    }

    for (0..num_cols) |col| {
        const header_cell = reader.getCell(0, col);
        const name = if (header_cell) |cell| blk: {
            const cell_name = xlsxCellToString(cell);
            break :blk if (cell_name) |n| try allocator.dupe(u8, n) else try std.fmt.allocPrint(allocator, "col_{d}", .{col});
        } else try std.fmt.allocPrint(allocator, "col_{d}", .{col});
        col_names[col] = name;

        const lance_type = inferXlsxColumnType(&reader, col);
        schema[col] = .{
            .name = name,
            .data_type = lance_type,
        };
        std.debug.print("    - {s}: {s}\n", .{name, @tagName(lance_type)});
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    std.debug.print("Converting columns...\n", .{});

    for (0..num_cols) |col| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        const lance_type = schema[col].data_type;

        switch (lance_type) {
            .float64 => {
                var values = try allocator.alloc(f64, num_data_rows);
                defer allocator.free(values);

                for (0..num_data_rows) |i| {
                    const row = i + 1;
                    if (reader.getCell(row, col)) |cell| {
                        values[i] = switch (cell) {
                            .number => |n| n,
                            else => 0.0,
                        };
                    } else {
                        values[i] = 0.0;
                    }
                }

                try encoder.writeFloat64Slice(values);
                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(col),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = null,
                });
            },
            .string => {
                var values = try allocator.alloc([]const u8, num_data_rows);
                defer allocator.free(values);

                for (0..num_data_rows) |i| {
                    const row = i + 1;
                    if (reader.getCell(row, col)) |cell| {
                        values[i] = xlsxCellToString(cell) orelse "";
                    } else {
                        values[i] = "";
                    }
                }

                try encoder.writeStrings(values, &offsets_buf, allocator);
                try lance_writer.writeColumnBatch(.{
                    .column_index = @intCast(col),
                    .data = encoder.getBytes(),
                    .row_count = @intCast(values.len),
                    .offsets = offsets_buf.items,
                });
            },
            else => {},
        }
    }

    try formats.finalizeLanceFile(&lance_writer, output_path);
}
