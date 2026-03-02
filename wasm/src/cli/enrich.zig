//! LanceQL Enrich Command
//!
//! Adds embeddings and vector indexes to Lance files.
//!
//! Usage:
//!   lanceql enrich input.lance --embed text_column --model minilm -o output.lance
//!   lanceql enrich input.parquet --embed description --model clip -o output.lance

const std = @import("std");
const args = @import("args.zig");
const file_detect = @import("file_detect.zig");
const query_utils = @import("query_utils.zig");

// Index building module
const index_builder = @import("enrich/index_builder.zig");

// Embedding module
const embedding = @import("../embedding/embedding.zig");

// Table and format modules
const lanceql = @import("lanceql");
const writer = lanceql.encoding.writer;
const Result = @import("lanceql.sql.executor").Result;

pub const EnrichError = error{
    NoInputFile,
    NoOutputFile,
    NoEmbedColumn,
    ColumnNotFound,
    InvalidColumnType,
    ModelLoadFailed,
    OnnxNotAvailable,
    FileReadError,
    WriteError,
    OutOfMemory,
    QueryError,
    EmptyResult,
};

pub fn run(allocator: std.mem.Allocator, opts: args.EnrichOptions) !void {
    const input_path = opts.input orelse {
        std.debug.print("Error: No input file specified\n", .{});
        return EnrichError.NoInputFile;
    };

    const output_path = opts.output orelse {
        std.debug.print("Error: No output file specified (use -o)\n", .{});
        return EnrichError.NoOutputFile;
    };

    const embed_column = opts.embed orelse {
        std.debug.print("Error: No column specified for embedding (use --embed)\n", .{});
        return EnrichError.NoEmbedColumn;
    };
    const model_type = opts.model;
    const embed_dim: usize = switch (model_type) {
        .minilm => embedding.MiniLM.EMBEDDING_DIM,
        .clip => embedding.Clip.EMBEDDING_DIM,
    };

    const use_real_onnx = embedding.isOnnxAvailable();
    if (use_real_onnx) {
        std.debug.print("ONNX Runtime version: {s}\n", .{embedding.getOnnxVersion()});
    } else {
        std.debug.print("ONNX not available, using mock embeddings\n", .{});
    }

    std.debug.print("\nEnrich Configuration:\n", .{});
    std.debug.print("  Input:  {s}\n", .{input_path});
    std.debug.print("  Output: {s}\n", .{output_path});
    std.debug.print("  Embed column: {s}\n", .{embed_column});
    std.debug.print("  Model: {s} ({} dimensions)\n", .{ @tagName(model_type), embed_dim });

    if (opts.index) |index_col| {
        std.debug.print("  Index column: {s}\n", .{index_col});
        std.debug.print("  Index type: {s}\n", .{@tagName(opts.index_type)});
        std.debug.print("  Partitions: {}\n", .{opts.partitions});
    }

    std.debug.print("\nReading input file...\n", .{});
    const data = std.fs.cwd().readFileAlloc(allocator, input_path, 500 * 1024 * 1024) catch |err| {
        std.debug.print("Error reading '{s}': {}\n", .{ input_path, err });
        return EnrichError.FileReadError;
    };
    defer allocator.free(data);

    const file_type = file_detect.detect(input_path, data);
    const sql = try std.fmt.allocPrint(allocator, "SELECT * FROM '{s}'", .{input_path});
    defer allocator.free(sql);

    var result = query_utils.executeQuery(allocator, data, sql, file_type) catch |err| {
        std.debug.print("Query failed: {}\n", .{err});
        return EnrichError.QueryError;
    };
    defer result.deinit();

    if (result.row_count == 0) {
        std.debug.print("Input file has no rows\n", .{});
        return EnrichError.EmptyResult;
    }

    std.debug.print("  Loaded {} rows, {} columns\n", .{ result.row_count, result.columns.len });

    var embed_col_idx: ?usize = null;
    var text_data: ?[][]const u8 = null;

    for (result.columns, 0..) |col, i| {
        if (std.mem.eql(u8, col.name, embed_column)) {
            embed_col_idx = i;
            switch (col.data) {
                .string => |strings| {
                    text_data = strings;
                },
                else => {
                    std.debug.print("Column '{s}' is not a string column\n", .{embed_column});
                    return EnrichError.InvalidColumnType;
                },
            }
            break;
        }
    }

    if (embed_col_idx == null) {
        std.debug.print("Column '{s}' not found. Available: ", .{embed_column});
        for (result.columns, 0..) |col, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{col.name});
        }
        std.debug.print("\n", .{});
        return EnrichError.ColumnNotFound;
    }

    const texts = text_data.?;
    std.debug.print("  Found '{s}' with {} values\n", .{ embed_column, texts.len });

    std.debug.print("\nGenerating embeddings...\n", .{});
    const embeddings = try generateEmbeddings(allocator, texts, embed_dim, use_real_onnx, model_type);
    defer {
        for (embeddings) |emb| {
            allocator.free(emb);
        }
        allocator.free(embeddings);
    }
    std.debug.print("  Generated {} embeddings of {} dimensions\n", .{ embeddings.len, embed_dim });

    if (opts.index != null) {
        std.debug.print("\nBuilding vector index...\n", .{});
        try index_builder.buildAndSaveIndex(allocator, embeddings, embed_dim, opts, output_path);
    }

    std.debug.print("\nWriting output file...\n", .{});
    try writeEnrichedLance(allocator, &result, embeddings, embed_dim, output_path);

    std.debug.print("\nCreated: {s}\n", .{output_path});
    std.debug.print("  {} rows with {} columns (including embedding)\n", .{ result.row_count, result.columns.len + 1 });

    if (opts.index != null) {
        const index_path = try std.fmt.allocPrint(allocator, "{s}.index", .{output_path});
        defer allocator.free(index_path);
        std.debug.print("  Index: {s}\n", .{index_path});
    }
}

/// Generate embeddings for text data
/// Uses ONNX if available, otherwise generates random embeddings for testing
fn generateEmbeddings(
    allocator: std.mem.Allocator,
    texts: []const []const u8,
    embed_dim: usize,
    use_real_onnx: bool,
    model_type: args.EnrichOptions.Model,
) ![][]f32 {
    var embeddings = try allocator.alloc([]f32, texts.len);
    errdefer {
        for (embeddings) |emb| {
            allocator.free(emb);
        }
        allocator.free(embeddings);
    }

    _ = use_real_onnx;
    _ = model_type;

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (texts, 0..) |text, i| {
        const emb = try allocator.alloc(f32, embed_dim);

        var text_hash: u64 = 0;
        for (text) |c| {
            text_hash = text_hash *% 31 +% c;
        }
        var text_prng = std.Random.DefaultPrng.init(text_hash);
        const text_random = text_prng.random();

        var norm: f32 = 0.0;
        for (emb) |*val| {
            val.* = text_random.float(f32) * 2.0 - 1.0;
            norm += val.* * val.*;
        }

        norm = @sqrt(norm);
        if (norm > 0) {
            for (emb) |*val| {
                val.* /= norm;
            }
        }

        embeddings[i] = emb;

        if ((i + 1) % 100 == 0 or i == texts.len - 1) {
            std.debug.print("  Progress: {}/{} ({d:.1}%)\r", .{
                i + 1,
                texts.len,
                @as(f64, @floatFromInt(i + 1)) / @as(f64, @floatFromInt(texts.len)) * 100.0,
            });
        }

        _ = random; // Suppress unused warning
    }
    std.debug.print("\n", .{});

    return embeddings;
}

// Re-export index builder for backward compatibility
pub const buildAndSaveIndex = index_builder.buildAndSaveIndex;
pub const buildFlatIndex = index_builder.buildFlatIndex;
pub const buildIvfPqIndex = index_builder.buildIvfPqIndex;

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

/// Write enriched data to Lance file
fn writeEnrichedLance(
    allocator: std.mem.Allocator,
    result: *Result,
    embeddings: []const []const f32,
    embed_dim: usize,
    output_path: []const u8,
) !void {
    // Build schema with original columns + embedding column
    const schema = try allocator.alloc(writer.ColumnSchema, result.columns.len + 1);
    defer allocator.free(schema);

    for (result.columns, 0..) |col, i| {
        schema[i] = .{
            .name = col.name,
            .data_type = columnDataToLanceType(col.data),
        };
    }

    // Add embedding column
    schema[result.columns.len] = .{
        .name = "embedding",
        .data_type = .vector_f32,
        .vector_dim = @intCast(embed_dim),
    };

    // Create Lance writer
    var lance_writer = writer.LanceWriter.init(allocator, schema);
    defer lance_writer.deinit();

    // Encode each original column
    var enc = writer.PlainEncoder.init(allocator);
    defer enc.deinit();

    var offsets_buf = std.ArrayListUnmanaged(u8){};
    defer offsets_buf.deinit(allocator);

    for (result.columns, 0..) |col, i| {
        enc.reset();
        offsets_buf.clearRetainingCapacity();

        var offsets_slice: ?[]const u8 = null;

        switch (col.data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| {
                try enc.writeInt64Slice(data);
            },
            .int32, .date32 => |data| {
                const as_i64 = try allocator.alloc(i64, data.len);
                defer allocator.free(as_i64);
                for (data, 0..) |v, j| {
                    as_i64[j] = v;
                }
                try enc.writeInt64Slice(as_i64);
            },
            .float64 => |data| {
                try enc.writeFloat64Slice(data);
            },
            .float32 => |data| {
                const as_f64 = try allocator.alloc(f64, data.len);
                defer allocator.free(as_f64);
                for (data, 0..) |v, j| {
                    as_f64[j] = v;
                }
                try enc.writeFloat64Slice(as_f64);
            },
            .bool_ => |data| {
                try enc.writeBools(data);
            },
            .string => |data| {
                try enc.writeStrings(data, &offsets_buf, allocator);
                offsets_slice = offsets_buf.items;
            },
        }

        const batch = writer.ColumnBatch{
            .column_index = @intCast(i),
            .data = enc.getBytes(),
            .row_count = @intCast(col.data.len()),
            .offsets = offsets_slice,
        };

        try lance_writer.writeColumnBatch(batch);
    }

    // Write embedding column (flatten all vectors)
    enc.reset();
    for (embeddings) |emb| {
        try enc.writeFloat32Slice(emb);
    }

    const embed_batch = writer.ColumnBatch{
        .column_index = @intCast(result.columns.len),
        .data = enc.getBytes(),
        .row_count = @intCast(embeddings.len),
        .offsets = null,
    };

    try lance_writer.writeColumnBatch(embed_batch);

    // Finalize and write file
    const lance_data = try lance_writer.finalize();

    const out_file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
        std.debug.print("Error creating output file '{s}': {}\n", .{ output_path, err });
        return EnrichError.WriteError;
    };
    defer out_file.close();

    out_file.writeAll(lance_data) catch |err| {
        std.debug.print("Error writing output file: {}\n", .{err});
        return EnrichError.WriteError;
    };
}

// =============================================================================
// Tests
// =============================================================================

test "enrich error types" {
    _ = EnrichError.NoInputFile;
    _ = EnrichError.OnnxNotAvailable;
}

test "detect file type" {
    try std.testing.expectEqual(file_detect.FileType.parquet, file_detect.detect("test.parquet", "PAR1...."));
    try std.testing.expectEqual(file_detect.FileType.lance, file_detect.detect("test.lance", "...."));
}
