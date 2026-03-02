//! LanceQL Transform Command
//!
//! Applies SQL-like transformations to data files and outputs to Lance format.
//!
//! Usage:
//!   lanceql transform input.parquet -o output.lance --select "col1,col2"
//!   lanceql transform data.lance -o filtered.lance --filter "x > 100"
//!   lanceql transform input.csv -o output.lance --rename "old:new" --limit 1000

const std = @import("std");
const lanceql = @import("lanceql");
const writer = lanceql.encoding.writer;
const Result = @import("lanceql.sql.executor").Result;
const args = @import("args.zig");
const file_detect = @import("file_detect.zig");
const query_utils = @import("query_utils.zig");

pub const TransformError = error{
    NoInputFile,
    NoOutputFile,
    FileNotFound,
    UnsupportedFormat,
    QueryError,
    WriteError,
    OutOfMemory,
};

/// Run the transform command
pub fn run(allocator: std.mem.Allocator, opts: args.TransformOptions) !void {
    // Validate input
    const input_path = opts.input orelse {
        std.debug.print("Error: No input file specified\n", .{});
        std.debug.print("Usage: lanceql transform <input> -o <output> [options]\n", .{});
        return TransformError.NoInputFile;
    };

    // Validate output
    const output_path = opts.output orelse {
        std.debug.print("Error: No output file specified (use -o)\n", .{});
        return TransformError.NoOutputFile;
    };

    // Build SQL query from options
    const sql = try buildSqlQuery(allocator, input_path, opts);
    defer allocator.free(sql);

    std.debug.print("Executing: {s}\n", .{sql});

    // Read input file
    const data = std.fs.cwd().readFileAlloc(allocator, input_path, 500 * 1024 * 1024) catch |err| {
        std.debug.print("Error reading '{s}': {}\n", .{ input_path, err });
        return TransformError.FileNotFound;
    };
    defer allocator.free(data);

    // Detect file type and execute query
    const file_type = file_detect.detect(input_path, data);
    var result = try query_utils.executeQuery(allocator, data, sql, file_type);
    defer result.deinit();

    std.debug.print("Query returned {d} rows, {d} columns\n", .{ result.row_count, result.columns.len });

    // Write result to Lance file
    try writeResultToLance(allocator, &result, output_path);
}

/// Build SQL query from transform options
fn buildSqlQuery(allocator: std.mem.Allocator, input_path: []const u8, opts: args.TransformOptions) ![]const u8 {
    var query = std.ArrayListUnmanaged(u8){};
    defer query.deinit(allocator);

    // SELECT clause
    try query.appendSlice(allocator, "SELECT ");

    if (opts.select) |select_cols| {
        // Handle rename: parse "old:new,a:b" format
        if (opts.rename) |rename_spec| {
            // Build select with AS aliases
            var col_iter = std.mem.splitScalar(u8, select_cols, ',');
            var first = true;

            while (col_iter.next()) |col| {
                if (!first) try query.appendSlice(allocator, ", ");
                first = false;

                const trimmed = std.mem.trim(u8, col, " ");

                // Check if this column has a rename
                const new_name = findRename(rename_spec, trimmed);
                if (new_name) |name| {
                    try query.appendSlice(allocator, trimmed);
                    try query.appendSlice(allocator, " AS ");
                    try query.appendSlice(allocator, name);
                } else {
                    try query.appendSlice(allocator, trimmed);
                }
            }
        } else {
            try query.appendSlice(allocator, select_cols);
        }
    } else if (opts.rename) |rename_spec| {
        // SELECT * with renames - need to expand
        // For now, just use * and note rename doesn't work without --select
        _ = rename_spec;
        try query.appendSlice(allocator, "*");
    } else {
        try query.appendSlice(allocator, "*");
    }

    // FROM clause
    try query.appendSlice(allocator, " FROM '");
    try query.appendSlice(allocator, input_path);
    try query.append(allocator, '\'');

    // WHERE clause
    if (opts.filter) |filter| {
        try query.appendSlice(allocator, " WHERE ");
        try query.appendSlice(allocator, filter);
    }

    // LIMIT clause
    if (opts.limit) |limit| {
        try query.appendSlice(allocator, " LIMIT ");
        try std.fmt.format(query.writer(allocator), "{d}", .{limit});
    }

    return try query.toOwnedSlice(allocator);
}

/// Find rename for a column: "old:new,a:b" -> "old" returns "new"
fn findRename(rename_spec: []const u8, col_name: []const u8) ?[]const u8 {
    var pair_iter = std.mem.splitScalar(u8, rename_spec, ',');
    while (pair_iter.next()) |pair| {
        const trimmed = std.mem.trim(u8, pair, " ");
        if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
            const old_name = trimmed[0..colon_pos];
            const new_name = trimmed[colon_pos + 1 ..];
            if (std.mem.eql(u8, old_name, col_name)) {
                return new_name;
            }
        }
    }
    return null;
}

/// Map Result.ColumnData type to writer.DataType
fn columnDataToLanceType(data: Result.ColumnData) writer.DataType {
    return switch (data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => .int64,
        .int32, .date32 => .int32,
        .float64 => .float64,
        .float32 => .float32,
        .bool_ => .bool,
        .string => .string,
    };
}

/// Write Result to Lance file
fn writeResultToLance(allocator: std.mem.Allocator, result: *Result, output_path: []const u8) !void {
    if (result.columns.len == 0) {
        std.debug.print("Warning: No columns in result, nothing to write\n", .{});
        return;
    }

    // Build Lance schema
    const schema = try allocator.alloc(writer.ColumnSchema, result.columns.len);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = columnDataToLanceType(col.data),
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

        switch (col.data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| {
                try encoder.writeInt64Slice(data);
            },
            .int32, .date32 => |data| {
                // Convert int32 to int64 for Lance
                const as_i64 = try allocator.alloc(i64, data.len);
                defer allocator.free(as_i64);
                for (data, 0..) |v, j| {
                    as_i64[j] = v;
                }
                try encoder.writeInt64Slice(as_i64);
            },
            .float64 => |data| {
                try encoder.writeFloat64Slice(data);
            },
            .float32 => |data| {
                // Convert float32 to float64 for Lance
                const as_f64 = try allocator.alloc(f64, data.len);
                defer allocator.free(as_f64);
                for (data, 0..) |v, j| {
                    as_f64[j] = v;
                }
                try encoder.writeFloat64Slice(as_f64);
            },
            .bool_ => |data| {
                try encoder.writeBools(data);
            },
            .string => |data| {
                try encoder.writeStrings(data, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = encoder.getBytes(),
            .row_count = @intCast(col.data.len()),
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    // Finalize and write file
    const lance_data = try lance_writer.finalize();

    // Write to output file
    const out_file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
        std.debug.print("Error creating output file '{s}': {}\n", .{ output_path, err });
        return TransformError.WriteError;
    };
    defer out_file.close();

    out_file.writeAll(lance_data) catch |err| {
        std.debug.print("Error writing output file: {}\n", .{err});
        return TransformError.WriteError;
    };

    std.debug.print("Created: {s} ({d} rows, {d} columns, {d} bytes)\n", .{
        output_path,
        result.row_count,
        result.columns.len,
        lance_data.len,
    });
}

// =============================================================================
// Tests
// =============================================================================

test "build sql query - select only" {
    const allocator = std.testing.allocator;
    const opts = args.TransformOptions{
        .input = "test.parquet",
        .output = "out.lance",
        .select = "id,name",
        .filter = null,
        .rename = null,
        .cast = null,
        .limit = null,
        .help = false,
    };

    const sql = try buildSqlQuery(allocator, "test.parquet", opts);
    defer allocator.free(sql);

    try std.testing.expectEqualStrings("SELECT id,name FROM 'test.parquet'", sql);
}

test "build sql query - with filter and limit" {
    const allocator = std.testing.allocator;
    const opts = args.TransformOptions{
        .input = "test.parquet",
        .output = "out.lance",
        .select = null,
        .filter = "x > 100",
        .rename = null,
        .cast = null,
        .limit = 50,
        .help = false,
    };

    const sql = try buildSqlQuery(allocator, "test.parquet", opts);
    defer allocator.free(sql);

    try std.testing.expectEqualStrings("SELECT * FROM 'test.parquet' WHERE x > 100 LIMIT 50", sql);
}

test "find rename" {
    try std.testing.expectEqualStrings("new_name", findRename("old:new_name", "old").?);
    try std.testing.expectEqualStrings("b", findRename("a:b,c:d", "a").?);
    try std.testing.expectEqualStrings("d", findRename("a:b,c:d", "c").?);
    try std.testing.expect(findRename("a:b", "x") == null);
}
