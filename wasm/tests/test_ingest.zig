//! Ingest Command Tests
//!
//! Tests for CSV, TSV, JSON, JSONL, and Parquet → Lance conversion.
//! Verifies that all ingest pipelines produce valid Lance files.

const std = @import("std");
const lanceql = @import("lanceql");
const encoding = @import("lanceql.encoding");
const format = @import("lanceql.format");
const parquet_enc = @import("lanceql.encoding.parquet");

const csv = encoding.csv;
const json = encoding.json;
const writer = encoding.writer;
const Footer = format.Footer;
const ParquetFile = format.ParquetFile;
const parquet_meta = format.parquet_metadata;
const PageReader = parquet_enc.PageReader;

// ============================================================================
// CSV Parser Tests
// ============================================================================

test "csv: parse simple CSV with header" {
    const allocator = std.testing.allocator;
    const data =
        \\id,name,value
        \\1,Alice,10.5
        \\2,Bob,20.5
        \\3,Charlie,30.5
    ;

    const result = try csv.readCsv(allocator, data, .{ .has_header = true });
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 3), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // Check column names
    try std.testing.expectEqualStrings("id", result.columns[0].name);
    try std.testing.expectEqualStrings("name", result.columns[1].name);
    try std.testing.expectEqualStrings("value", result.columns[2].name);

    // Check types
    try std.testing.expectEqual(csv.ColumnType.int64, result.columns[0].col_type);
    try std.testing.expectEqual(csv.ColumnType.string, result.columns[1].col_type);
    try std.testing.expectEqual(csv.ColumnType.float64, result.columns[2].col_type);

    // Check values
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].int64_values.items[0]);
    try std.testing.expectEqual(@as(i64, 2), result.columns[0].int64_values.items[1]);
    try std.testing.expectEqual(@as(i64, 3), result.columns[0].int64_values.items[2]);

    try std.testing.expectEqualStrings("Alice", result.columns[1].string_values.items[0]);
    try std.testing.expectEqualStrings("Bob", result.columns[1].string_values.items[1]);
    try std.testing.expectEqualStrings("Charlie", result.columns[1].string_values.items[2]);

    try std.testing.expectApproxEqAbs(@as(f64, 10.5), result.columns[2].float64_values.items[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 20.5), result.columns[2].float64_values.items[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 30.5), result.columns[2].float64_values.items[2], 0.01);
}

test "csv: parse TSV format" {
    const allocator = std.testing.allocator;
    const data = "id\tname\tactive\n1\tAlice\ttrue\n2\tBob\tfalse";

    const result = try csv.readCsv(allocator, data, .{ .delimiter = '\t', .has_header = true });
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 3), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    // Check boolean type inference
    try std.testing.expectEqual(csv.ColumnType.bool_, result.columns[2].col_type);
    try std.testing.expectEqual(true, result.columns[2].bool_values.items[0]);
    try std.testing.expectEqual(false, result.columns[2].bool_values.items[1]);
}

test "csv: type inference with mixed values" {
    const allocator = std.testing.allocator;
    const data =
        \\col
        \\123
        \\456.78
        \\text
    ;

    const result = try csv.readCsv(allocator, data, .{ .has_header = true });
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Mixed types should fall back to string
    try std.testing.expectEqual(csv.ColumnType.string, result.columns[0].col_type);
}

test "csv: delimiter auto-detection" {
    try std.testing.expectEqual(@as(u8, ','), csv.detectDelimiter("a,b,c\n1,2,3"));
    try std.testing.expectEqual(@as(u8, '\t'), csv.detectDelimiter("a\tb\tc\n1\t2\t3"));
    try std.testing.expectEqual(@as(u8, ';'), csv.detectDelimiter("a;b;c\n1;2;3"));
    try std.testing.expectEqual(@as(u8, '|'), csv.detectDelimiter("a|b|c\n1|2|3"));
}

// ============================================================================
// JSON/JSONL Parser Tests
// ============================================================================

test "json: detect format - JSON array" {
    try std.testing.expectEqual(json.Format.json_array, json.detectFormat("[{\"a\": 1}]"));
    try std.testing.expectEqual(json.Format.json_array, json.detectFormat("  [{\"a\": 1}]"));
    try std.testing.expectEqual(json.Format.json_array, json.detectFormat("\n[{\"a\": 1}]"));
}

test "json: detect format - JSONL" {
    try std.testing.expectEqual(json.Format.jsonl, json.detectFormat("{\"a\": 1}"));
    try std.testing.expectEqual(json.Format.jsonl, json.detectFormat("  {\"a\": 1}"));
    try std.testing.expectEqual(json.Format.jsonl, json.detectFormat("{\"a\": 1}\n{\"a\": 2}"));
}

test "json: detect format - empty and unknown" {
    try std.testing.expectEqual(json.Format.empty, json.detectFormat(""));
    try std.testing.expectEqual(json.Format.empty, json.detectFormat("   "));
    try std.testing.expectEqual(json.Format.unknown, json.detectFormat("abc"));
}

test "json: parse JSON array with multiple rows" {
    const allocator = std.testing.allocator;
    const data =
        \\[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    // Check column types
    try std.testing.expectEqual(csv.ColumnType.int64, result.columns[0].col_type);
    try std.testing.expectEqual(csv.ColumnType.string, result.columns[1].col_type);

    // Check values
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].int64_values.items[0]);
    try std.testing.expectEqual(@as(i64, 2), result.columns[0].int64_values.items[1]);
    try std.testing.expectEqualStrings("Alice", result.columns[1].string_values.items[0]);
    try std.testing.expectEqualStrings("Bob", result.columns[1].string_values.items[1]);
}

test "json: parse JSONL format" {
    const allocator = std.testing.allocator;
    const data =
        \\{"name": "Alice", "age": 30}
        \\{"name": "Bob", "age": 25}
        \\{"name": "Charlie", "age": 35}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // Verify string column has correct values
    const name_col = for (result.columns) |col| {
        if (std.mem.eql(u8, col.name, "name")) break col;
    } else unreachable;

    try std.testing.expectEqualStrings("Alice", name_col.string_values.items[0]);
    try std.testing.expectEqualStrings("Bob", name_col.string_values.items[1]);
    try std.testing.expectEqualStrings("Charlie", name_col.string_values.items[2]);
}

test "json: type inference int64" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1}
        \\{"value": 2}
        \\{"value": 3}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(csv.ColumnType.int64, result.columns[0].col_type);
    try std.testing.expectEqual(@as(i64, 1), result.columns[0].int64_values.items[0]);
    try std.testing.expectEqual(@as(i64, 2), result.columns[0].int64_values.items[1]);
    try std.testing.expectEqual(@as(i64, 3), result.columns[0].int64_values.items[2]);
}

test "json: type inference float64" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1.5}
        \\{"value": 2.5}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(csv.ColumnType.float64, result.columns[0].col_type);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), result.columns[0].float64_values.items[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), result.columns[0].float64_values.items[1], 0.01);
}

test "json: type inference bool" {
    const allocator = std.testing.allocator;
    const data =
        \\{"active": true}
        \\{"active": false}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(csv.ColumnType.bool_, result.columns[0].col_type);
    try std.testing.expectEqual(true, result.columns[0].bool_values.items[0]);
    try std.testing.expectEqual(false, result.columns[0].bool_values.items[1]);
}

test "json: type promotion int to float" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1}
        \\{"value": 2.5}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Should promote to float64
    try std.testing.expectEqual(csv.ColumnType.float64, result.columns[0].col_type);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.columns[0].float64_values.items[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), result.columns[0].float64_values.items[1], 0.01);
}

test "json: mixed types fallback to string" {
    const allocator = std.testing.allocator;
    const data =
        \\{"value": 1}
        \\{"value": "text"}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Incompatible types should fallback to string
    try std.testing.expectEqual(csv.ColumnType.string, result.columns[0].col_type);
}

test "json: nested object flattening with dot notation" {
    const allocator = std.testing.allocator;
    const data =
        \\{"user": {"name": "Alice", "profile": {"level": 5}}}
        \\{"user": {"name": "Bob", "profile": {"level": 10}}}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Should have flattened columns: user.name, user.profile.level
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);

    // Find columns by name
    var found_name = false;
    var found_level = false;
    for (result.columns) |col| {
        if (std.mem.eql(u8, col.name, "user.name")) {
            found_name = true;
            try std.testing.expectEqual(csv.ColumnType.string, col.col_type);
            try std.testing.expectEqualStrings("Alice", col.string_values.items[0]);
            try std.testing.expectEqualStrings("Bob", col.string_values.items[1]);
        }
        if (std.mem.eql(u8, col.name, "user.profile.level")) {
            found_level = true;
            try std.testing.expectEqual(csv.ColumnType.int64, col.col_type);
            try std.testing.expectEqual(@as(i64, 5), col.int64_values.items[0]);
            try std.testing.expectEqual(@as(i64, 10), col.int64_values.items[1]);
        }
    }
    try std.testing.expect(found_name);
    try std.testing.expect(found_level);
}

test "json: sparse columns (missing keys in some rows)" {
    const allocator = std.testing.allocator;
    const data =
        \\{"a": 1, "b": 2}
        \\{"a": 3}
        \\{"a": 5, "b": 6}
    ;

    const result = try json.readJson(allocator, data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 3), result.row_count);

    // Find column 'b' and check it has 3 values (with default for missing)
    for (result.columns) |col| {
        if (std.mem.eql(u8, col.name, "b")) {
            try std.testing.expectEqual(@as(usize, 3), col.int64_values.items.len);
            try std.testing.expectEqual(@as(i64, 2), col.int64_values.items[0]);
            try std.testing.expectEqual(@as(i64, 0), col.int64_values.items[1]); // Default for null
            try std.testing.expectEqual(@as(i64, 6), col.int64_values.items[2]);
        }
    }
}

// ============================================================================
// Lance Writer Tests
// ============================================================================

test "writer: encode int64 column" {
    const allocator = std.testing.allocator;

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    const values = [_]i64{ 1, 2, 3, 4, 5 };
    try encoder.writeInt64Slice(&values);

    const bytes = encoder.getBytes();
    try std.testing.expectEqual(@as(usize, 5 * 8), bytes.len);

    // Verify little-endian encoding
    const aligned: []align(8) const u8 = @alignCast(bytes);
    const decoded = std.mem.bytesAsSlice(i64, aligned);
    try std.testing.expectEqual(@as(i64, 1), decoded[0]);
    try std.testing.expectEqual(@as(i64, 5), decoded[4]);
}

test "writer: encode float64 column" {
    const allocator = std.testing.allocator;

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    const values = [_]f64{ 1.5, 2.5, 3.5 };
    try encoder.writeFloat64Slice(&values);

    const bytes = encoder.getBytes();
    try std.testing.expectEqual(@as(usize, 3 * 8), bytes.len);
}

test "writer: encode string column with offsets" {
    const allocator = std.testing.allocator;

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    const values = [_][]const u8{ "hello", "world", "test" };
    try encoder.writeStrings(&values, &offsets_buf, allocator);

    // Check data bytes contain concatenated strings
    const bytes = encoder.getBytes();
    try std.testing.expectEqualStrings("helloworldtest", bytes);

    // Check offsets
    try std.testing.expect(offsets_buf.items.len > 0);
}

test "writer: encode bool column" {
    const allocator = std.testing.allocator;

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    const values = [_]bool{ true, false, true, true, false };
    try encoder.writeBools(&values);

    const bytes = encoder.getBytes();
    // Bools are packed as bits, so 5 bools = 1 byte
    try std.testing.expectEqual(@as(usize, 1), bytes.len);

    // Check bit pattern: true=1, false=0
    // First bool is least significant bit
    // Pattern: true(1), false(0), true(1), true(1), false(0) = 0b01101 = 13
    try std.testing.expectEqual(@as(u8, 0b01101), bytes[0]);
}

// ============================================================================
// End-to-End Ingest Tests
// ============================================================================

test "e2e: CSV to Lance round-trip" {
    const allocator = std.testing.allocator;

    // Parse CSV
    const csv_data =
        \\id,name,score
        \\1,Alice,95.5
        \\2,Bob,87.3
        \\3,Charlie,92.1
    ;

    const result = try csv.readCsv(allocator, csv_data, .{ .has_header = true });
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Build schema
    var schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = switch (col.col_type) {
                .int64 => .int64,
                .float64 => .float64,
                .bool_ => .bool,
                .string => .string,
            },
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each column
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

    // Finalize
    const lance_data = try lance_writer.finalize();
    try std.testing.expect(lance_data.len > 40); // At least footer size

    // Verify footer magic
    const footer_start = lance_data.len - 4;
    try std.testing.expectEqualStrings("LANC", lance_data[footer_start..]);

    // Parse footer
    const footer = try Footer.parse(lance_data[lance_data.len - 40 ..][0..40]);
    try std.testing.expectEqual(@as(u32, 3), footer.num_columns);
}

test "e2e: JSON to Lance round-trip" {
    const allocator = std.testing.allocator;

    // Parse JSON
    const json_data =
        \\[{"id": 1, "value": 100}, {"id": 2, "value": 200}]
    ;

    const result = try json.readJson(allocator, json_data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Build schema
    var schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = switch (col.col_type) {
                .int64 => .int64,
                .float64 => .float64,
                .bool_ => .bool,
                .string => .string,
            },
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    for (result.columns, 0..) |col, i| {
        encoder.reset();

        switch (col.col_type) {
            .int64 => try encoder.writeInt64Slice(col.int64_values.items),
            .float64 => try encoder.writeFloat64Slice(col.float64_values.items),
            else => {},
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = encoder.getBytes(),
            .row_count = @intCast(col.len()),
            .offsets = null,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    const lance_data = try lance_writer.finalize();

    // Verify footer
    try std.testing.expectEqualStrings("LANC", lance_data[lance_data.len - 4 ..]);
    const footer = try Footer.parse(lance_data[lance_data.len - 40 ..][0..40]);
    try std.testing.expectEqual(@as(u32, 2), footer.num_columns);
}

test "e2e: JSONL to Lance with nested objects" {
    const allocator = std.testing.allocator;

    const jsonl_data =
        \\{"user": {"name": "Alice"}, "score": 100}
        \\{"user": {"name": "Bob"}, "score": 200}
    ;

    const result = try json.readJson(allocator, jsonl_data, .{});
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    // Verify nested flattening
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);

    var found_user_name = false;
    var found_score = false;
    for (result.columns) |col| {
        if (std.mem.eql(u8, col.name, "user.name")) found_user_name = true;
        if (std.mem.eql(u8, col.name, "score")) found_score = true;
    }
    try std.testing.expect(found_user_name);
    try std.testing.expect(found_score);

    // Build and write to Lance format
    var schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = switch (col.col_type) {
                .int64 => .int64,
                .float64 => .float64,
                .bool_ => .bool,
                .string => .string,
            },
        };
    }

    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

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

    const lance_data = try lance_writer.finalize();
    try std.testing.expectEqualStrings("LANC", lance_data[lance_data.len - 4 ..]);
}

// ============================================================================
// Type Promotion Tests
// ============================================================================

test "json type promotion: null + int -> int" {
    try std.testing.expectEqual(json.ColumnType.int64, json.ColumnType.null_.promote(.int64));
    try std.testing.expectEqual(json.ColumnType.int64, json.ColumnType.int64.promote(.null_));
}

test "json type promotion: int + float -> float" {
    try std.testing.expectEqual(json.ColumnType.float64, json.ColumnType.int64.promote(.float64));
    try std.testing.expectEqual(json.ColumnType.float64, json.ColumnType.float64.promote(.int64));
}

test "json type promotion: incompatible -> string" {
    try std.testing.expectEqual(json.ColumnType.string, json.ColumnType.int64.promote(.string));
    try std.testing.expectEqual(json.ColumnType.string, json.ColumnType.bool_.promote(.int64));
    try std.testing.expectEqual(json.ColumnType.string, json.ColumnType.float64.promote(.bool_));
}

// ============================================================================
// Edge Cases
// ============================================================================

test "csv: empty input" {
    const allocator = std.testing.allocator;
    const result = csv.readCsv(allocator, "", .{});
    try std.testing.expectError(error.EmptyInput, result);
}

test "json: empty input" {
    const allocator = std.testing.allocator;
    const result = json.readJson(allocator, "", .{});
    try std.testing.expectError(json.JsonError.EmptyInput, result);
}

test "json: invalid format" {
    const allocator = std.testing.allocator;
    const result = json.readJson(allocator, "not json", .{});
    try std.testing.expectError(json.JsonError.InvalidJson, result);
}

test "csv: quoted fields with commas" {
    const allocator = std.testing.allocator;
    const data =
        \\name,description
        \\"John","Hello, World"
        \\"Jane","Test, with, commas"
    ;

    const result = try csv.readCsv(allocator, data, .{ .has_header = true });
    defer {
        for (result.columns) |*col| col.deinit();
        allocator.free(result.columns);
    }

    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 2), result.row_count);
    try std.testing.expectEqualStrings("Hello, World", result.columns[1].string_values.items[0]);
    try std.testing.expectEqualStrings("Test, with, commas", result.columns[1].string_values.items[1]);
}

// ============================================================================
// Parquet Ingestion Tests
// ============================================================================

/// Helper to map Parquet types to Lance types (mirrors ingest.zig)
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

test "parquet: type mapping verification" {
    // Test all Parquet type → Lance type mappings
    try std.testing.expectEqual(writer.DataType.bool, parquetTypeToLanceType(.boolean));
    try std.testing.expectEqual(writer.DataType.int32, parquetTypeToLanceType(.int32));
    try std.testing.expectEqual(writer.DataType.int64, parquetTypeToLanceType(.int64));
    try std.testing.expectEqual(writer.DataType.float32, parquetTypeToLanceType(.float));
    try std.testing.expectEqual(writer.DataType.float64, parquetTypeToLanceType(.double));
    try std.testing.expectEqual(writer.DataType.string, parquetTypeToLanceType(.byte_array));
    try std.testing.expectEqual(writer.DataType.string, parquetTypeToLanceType(.fixed_len_byte_array));
    try std.testing.expectEqual(writer.DataType.int64, parquetTypeToLanceType(.int96));
}

test "e2e: Parquet to Lance round-trip" {
    const allocator = std.testing.allocator;

    // Read Parquet fixture
    const file = std.fs.cwd().openFile("tests/fixtures/simple.parquet", .{}) catch |err| {
        std.debug.print("Could not open test file: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 2 * 1024 * 1024) catch |err| {
        std.debug.print("Could not read test file: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer allocator.free(data);

    // Parse Parquet
    var pf = try ParquetFile.init(allocator, data);
    defer pf.deinit();

    const num_cols = pf.getNumColumns();
    const num_rows = @as(usize, @intCast(pf.getNumRows()));

    // Verify Parquet metadata
    try std.testing.expectEqual(@as(usize, 3), num_cols);
    try std.testing.expectEqual(@as(usize, 5), num_rows);

    // Get column names
    const col_names = try pf.getColumnNames();
    defer allocator.free(col_names);

    // Build Lance schema
    var schema = try allocator.alloc(writer.ColumnSchema, num_cols);
    defer allocator.free(schema);

    const rg = pf.getRowGroup(0) orelse return error.SkipZigTest;
    for (0..num_cols) |i| {
        const col = rg.columns[i];
        const col_meta = col.meta_data orelse continue;
        schema[i] = .{
            .name = col_names[i],
            .data_type = parquetTypeToLanceType(col_meta.type_),
        };
    }

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    var encoder = writer.PlainEncoder.init(allocator);
    defer encoder.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    // Encode each column
    for (0..num_cols) |col_idx| {
        encoder.reset();
        offsets_buf.clearRetainingCapacity();

        const col = rg.columns[col_idx];
        const col_meta = col.meta_data orelse continue;
        const col_data = pf.getColumnData(0, col_idx) orelse continue;

        var page_reader = PageReader.init(
            col_data,
            col_meta.type_,
            null,
            col_meta.codec,
            allocator,
        );
        defer page_reader.deinit();

        var page = page_reader.readAll() catch continue;
        defer page.deinit(allocator);

        var row_count: u32 = @intCast(page.num_values);
        var offsets_slice: ?[]const u8 = null;

        switch (schema[col_idx].data_type) {
            .int64 => {
                if (page.int64_values) |values| {
                    try encoder.writeInt64Slice(values);
                    row_count = @intCast(values.len);
                }
            },
            .float64 => {
                if (page.double_values) |values| {
                    try encoder.writeFloat64Slice(values);
                    row_count = @intCast(values.len);
                }
            },
            .string => {
                if (page.binary_values) |values| {
                    try encoder.writeStrings(values, &offsets_buf, allocator);
                    offsets_slice = offsets_buf.items;
                    row_count = @intCast(values.len);
                }
            },
            else => continue,
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(col_idx),
            .data = encoder.getBytes(),
            .row_count = row_count,
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    // Finalize and verify
    const lance_data = try lance_writer.finalize();
    try std.testing.expect(lance_data.len > 40);

    // Verify footer magic
    try std.testing.expectEqualStrings("LANC", lance_data[lance_data.len - 4 ..]);

    // Parse and verify footer
    const footer = try Footer.parse(lance_data[lance_data.len - 40 ..][0..40]);
    try std.testing.expectEqual(@as(u32, 3), footer.num_columns);
}

test "e2e: Parquet with PLAIN encoding to Lance" {
    const allocator = std.testing.allocator;

    // Read PLAIN-encoded Parquet (no dictionary)
    const file = std.fs.cwd().openFile("tests/fixtures/simple_plain.parquet", .{}) catch |err| {
        std.debug.print("Could not open test file: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 2 * 1024 * 1024) catch |err| {
        std.debug.print("Could not read test file: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer allocator.free(data);

    var pf = try ParquetFile.init(allocator, data);
    defer pf.deinit();

    // Verify metadata
    try std.testing.expectEqual(@as(usize, 3), pf.getNumColumns());
    try std.testing.expectEqual(@as(i64, 5), pf.getNumRows());

    // Read and verify column data through PageReader
    const rg = pf.getRowGroup(0) orelse return error.SkipZigTest;

    // Read INT64 column (id)
    {
        const col = rg.columns[0];
        const col_meta = col.meta_data orelse return error.SkipZigTest;
        const col_data = pf.getColumnData(0, 0) orelse return error.SkipZigTest;

        var reader = PageReader.init(col_data, col_meta.type_, null, col_meta.codec, allocator);
        defer reader.deinit();

        var page = try reader.readAll();
        defer page.deinit(allocator);

        if (page.int64_values) |values| {
            try std.testing.expectEqual(@as(usize, 5), values.len);
            try std.testing.expectEqual(@as(i64, 1), values[0]);
            try std.testing.expectEqual(@as(i64, 5), values[4]);
        }
    }

    // Read DOUBLE column (value)
    {
        const col = rg.columns[2];
        const col_meta = col.meta_data orelse return error.SkipZigTest;
        const col_data = pf.getColumnData(0, 2) orelse return error.SkipZigTest;

        var reader = PageReader.init(col_data, col_meta.type_, null, col_meta.codec, allocator);
        defer reader.deinit();

        var page = try reader.readAll();
        defer page.deinit(allocator);

        if (page.double_values) |values| {
            try std.testing.expectEqual(@as(usize, 5), values.len);
            try std.testing.expectApproxEqAbs(@as(f64, 1.1), values[0], 0.001);
            try std.testing.expectApproxEqAbs(@as(f64, 5.5), values[4], 0.001);
        }
    }
}

test "e2e: Parquet with Snappy compression to Lance" {
    const allocator = std.testing.allocator;

    // Read Snappy-compressed Parquet
    const file = std.fs.cwd().openFile("tests/fixtures/simple_snappy.parquet", .{}) catch |err| {
        std.debug.print("Could not open test file: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 2 * 1024 * 1024) catch |err| {
        std.debug.print("Could not read test file: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer allocator.free(data);

    var pf = try ParquetFile.init(allocator, data);
    defer pf.deinit();

    // Verify metadata
    try std.testing.expectEqual(@as(usize, 3), pf.getNumColumns());
    try std.testing.expectEqual(@as(i64, 5), pf.getNumRows());

    const rg = pf.getRowGroup(0) orelse return error.SkipZigTest;

    // Verify compression codec is Snappy
    const col = rg.columns[0];
    const col_meta = col.meta_data orelse return error.SkipZigTest;
    try std.testing.expectEqual(parquet_meta.CompressionCodec.snappy, col_meta.codec);

    // Read decompressed data
    const col_data = pf.getColumnData(0, 0) orelse return error.SkipZigTest;

    var reader = PageReader.init(col_data, col_meta.type_, null, col_meta.codec, allocator);
    defer reader.deinit();

    var page = try reader.readAll();
    defer page.deinit(allocator);

    // Verify data decompressed correctly
    if (page.int64_values) |values| {
        try std.testing.expectEqual(@as(usize, 5), values.len);
        try std.testing.expectEqual(@as(i64, 1), values[0]);
        try std.testing.expectEqual(@as(i64, 2), values[1]);
        try std.testing.expectEqual(@as(i64, 3), values[2]);
        try std.testing.expectEqual(@as(i64, 4), values[3]);
        try std.testing.expectEqual(@as(i64, 5), values[4]);
    }
}
