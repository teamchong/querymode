//! LanceQL CLI - High-performance data pipeline for Lance files
//!
//! Usage:
//!   lanceql query "SELECT * FROM 'data.lance' LIMIT 10"
//!   lanceql ingest data.csv -o out.lance
//!   lanceql transform data.lance --select "a,b"
//!   lanceql enrich data.lance --embed text
//!   lanceql serve data.lance
//!   lanceql (no args) - auto-detect config or serve
//!
//! Designed for apple-to-apple comparison with:
//!   duckdb -c "SELECT * FROM 'data.parquet' LIMIT 10"
//!   polars -c "SELECT * FROM read_parquet('data.parquet') LIMIT 10"

const std = @import("std");
const args = @import("cli/args.zig");
const ingest = @import("cli/ingest.zig");
const enrich = @import("cli/enrich.zig");
const transform = @import("cli/transform.zig");
const serve = @import("cli/serve.zig");
const output = @import("cli/output.zig");
const benchmark = @import("cli/benchmark.zig");
const file_utils = @import("cli/file_utils.zig");
const yaml = @import("cli/yaml.zig");
const file_detect = @import("cli/file_detect.zig");
const lanceql = @import("lanceql");
const metal = @import("lanceql.gpu");
const Table = @import("lanceql.table").Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const DeltaTable = @import("lanceql.delta_table").DeltaTable;
const IcebergTable = @import("lanceql.iceberg_table").IcebergTable;
const ArrowTable = @import("lanceql.arrow_table").ArrowTable;
const AvroTable = @import("lanceql.avro_table").AvroTable;
const OrcTable = @import("lanceql.orc_table").OrcTable;
const XlsxTable = @import("lanceql.xlsx_table").XlsxTable;
const AnyTable = @import("lanceql.any_table").AnyTable;
const executor = @import("lanceql.sql.executor");
const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const ast = @import("lanceql.sql.ast");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const parsed = try args.parse(allocator);

    // Handle global commands
    switch (parsed.command) {
        .version => {
            std.debug.print("lanceql {s}\n", .{args.version});
            return;
        },
        .help => {
            args.printHelp();
            return;
        },
        .query => {
            if (parsed.global.help or parsed.query.help) {
                args.printQueryHelp();
                return;
            }
            try cmdQuery(allocator, parsed.query);
        },
        .ingest => {
            if (parsed.global.help or parsed.ingest.help) {
                args.printIngestHelp();
                return;
            }
            try cmdIngest(allocator, parsed.ingest);
        },
        .transform => {
            if (parsed.global.help or parsed.transform.help) {
                args.printTransformHelp();
                return;
            }
            try cmdTransform(allocator, parsed.transform);
        },
        .enrich => {
            if (parsed.global.help or parsed.enrich.help) {
                args.printEnrichHelp();
                return;
            }
            try cmdEnrich(allocator, parsed.enrich);
        },
        .serve => {
            if (parsed.global.help or parsed.serve.help) {
                args.printServeHelp();
                return;
            }
            try cmdServe(allocator, parsed.serve);
        },
        .history => {
            if (parsed.global.help or parsed.history.help) {
                args.printHistoryHelp();
                return;
            }
            try cmdHistory(allocator, parsed.history);
        },
        .diff => {
            if (parsed.global.help or parsed.diff_opts.help) {
                args.printDiffHelp();
                return;
            }
            try cmdDiff(allocator, parsed.diff_opts);
        },
        .none => {
            // No command - auto-detect mode
            // Check for config file or start serve
            if (parsed.global.config) |config_path| {
                try runConfigFile(allocator, config_path);
            } else if (findConfigFile()) |config_path| {
                try runConfigFile(allocator, config_path);
            } else {
                // Default to help
                args.printHelp();
            }
        },
    }
}

/// Query command - execute SQL on Lance/Parquet files
fn cmdQuery(allocator: std.mem.Allocator, opts: args.QueryOptions) !void {
    // Read query from file if specified
    var query_text = opts.query;
    var file_content: ?[]const u8 = null;

    if (opts.file) |file_path| {
        const f = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            std.debug.print("Error opening file '{s}': {}\n", .{ file_path, err });
            return;
        };
        defer f.close();

        file_content = f.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
            std.debug.print("Error reading file: {}\n", .{err});
            return;
        };
        query_text = file_content;
    }

    defer if (file_content) |fc| allocator.free(fc);

    if (query_text == null) {
        args.printQueryHelp();
        return;
    }

    // Initialize GPU if available
    _ = metal.initGPU();
    defer metal.cleanupGPU();

    const query = query_text.?;

    // Convert QueryOptions to legacy Args for existing functions
    const legacy_args = LegacyArgs{
        .query = query,
        .benchmark = opts.benchmark,
        .iterations = opts.iterations,
        .warmup = opts.warmup,
        .json = opts.json,
        .csv = opts.csv,
    };

    if (opts.benchmark) {
        try benchmark.run(allocator, query, .{
            .iterations = opts.iterations,
            .warmup = opts.warmup,
            .json = opts.json,
        });
    } else {
        try runQuery(allocator, query, legacy_args);
    }
}

/// Ingest command - convert CSV/JSON/Parquet to Lance
fn cmdIngest(allocator: std.mem.Allocator, opts: args.IngestOptions) !void {
    if (opts.input == null) {
        if (std.posix.isatty(std.posix.STDIN_FILENO)) {
            try serve.runConfigMode(allocator, "ingest");
            return;
        }
        args.printIngestHelp();
        return;
    }
    try ingest.run(allocator, opts);
}

/// Transform command - apply transformations to data files
fn cmdTransform(allocator: std.mem.Allocator, opts: args.TransformOptions) !void {
    if (opts.input == null) {
        if (std.posix.isatty(std.posix.STDIN_FILENO)) {
            try serve.runConfigMode(allocator, "transform");
            return;
        }
        args.printTransformHelp();
        return;
    }
    transform.run(allocator, opts) catch |err| {
        std.debug.print("Transform command failed: {}\n", .{err});
        return err;
    };
}

/// Enrich command - add embeddings and indexes
fn cmdEnrich(allocator: std.mem.Allocator, opts: args.EnrichOptions) !void {
    if (opts.input == null) {
        if (std.posix.isatty(std.posix.STDIN_FILENO)) {
            try serve.runConfigMode(allocator, "enrich");
            return;
        }
        args.printEnrichHelp();
        return;
    }
    enrich.run(allocator, opts) catch |err| {
        std.debug.print("Enrich command failed: {}\n", .{err});
        return err;
    };
}

/// Serve command - start interactive web server
fn cmdServe(allocator: std.mem.Allocator, opts: args.ServeOptions) !void {
    serve.run(allocator, opts) catch |err| {
        std.debug.print("Serve command failed: {}\n", .{err});
        return err;
    };
}

/// History command - show version history for a Lance table
fn cmdHistory(allocator: std.mem.Allocator, opts: args.HistoryOptions) !void {
    const table_path = opts.input orelse {
        args.printHistoryHelp();
        return;
    };

    // Load the Lance file
    const data = file_utils.openFileOrDataset(allocator, table_path) orelse {
        std.debug.print("Error opening '{s}': file not found or unreadable\n", .{table_path});
        return;
    };
    defer allocator.free(data);

    // Initialize table
    var table = Table.init(allocator, data) catch |err| {
        std.debug.print("Error loading Lance file '{s}': {}\n", .{ table_path, err });
        return;
    };
    defer table.deinit();

    // Create executor with dataset path for manifest access
    var exec = executor.Executor.init(&table, allocator);
    defer exec.deinit();
    exec.setDatasetPath(table_path);

    // Build and parse SQL query
    var sql_buf: [1024]u8 = undefined;
    const sql = if (opts.limit) |limit|
        std.fmt.bufPrint(&sql_buf, "SHOW VERSIONS FOR t LIMIT {d}", .{limit}) catch {
            std.debug.print("Error: Query too long\n", .{});
            return;
        }
    else
        std.fmt.bufPrint(&sql_buf, "SHOW VERSIONS FOR t", .{}) catch {
            std.debug.print("Error: Query too long\n", .{});
            return;
        };

    // Parse and execute
    const parsed = tokenizeAndParse(allocator, sql) catch |err| {
        std.debug.print("Parse error: {}\n", .{err});
        return;
    };
    var tokens = parsed.tokens;
    defer tokens.deinit(allocator);

    const stmt_result = exec.executeStatement(&parsed.stmt, &[_]ast.Value{}) catch |err| {
        std.debug.print("Execution error: {}\n", .{err});
        return;
    };

    // Output results
    switch (stmt_result) {
        .versions_list => |versions| {
            if (opts.json) {
                outputVersionsJson(versions);
            } else {
                outputVersionsTable(versions);
            }
            allocator.free(versions);
        },
        else => {
            std.debug.print("Unexpected result type\n", .{});
        },
    }
}

/// Diff command - show differences between versions
fn cmdDiff(allocator: std.mem.Allocator, opts: args.DiffOptions) !void {
    const table_path = opts.input orelse {
        args.printDiffHelp();
        return;
    };

    const from_version = opts.from orelse {
        std.debug.print("Error: --from version is required\n\n", .{});
        args.printDiffHelp();
        return;
    };

    // Load the Lance file
    const data = file_utils.openFileOrDataset(allocator, table_path) orelse {
        std.debug.print("Error opening '{s}': file not found or unreadable\n", .{table_path});
        return;
    };
    defer allocator.free(data);

    // Initialize table
    var table = Table.init(allocator, data) catch |err| {
        std.debug.print("Error loading Lance file '{s}': {}\n", .{ table_path, err });
        return;
    };
    defer table.deinit();

    // Create executor with dataset path for manifest access
    var exec = executor.Executor.init(&table, allocator);
    defer exec.deinit();
    exec.setDatasetPath(table_path);

    // Build SQL query
    var sql_buf: [1024]u8 = undefined;
    const limit = opts.limit orelse 100;

    const sql = if (opts.to) |to_version|
        std.fmt.bufPrint(&sql_buf, "DIFF t VERSION {d} AND VERSION {d} LIMIT {d}", .{ from_version, to_version, limit }) catch {
            std.debug.print("Error: Query too long\n", .{});
            return;
        }
    else
        std.fmt.bufPrint(&sql_buf, "DIFF t VERSION {d} AND VERSION HEAD LIMIT {d}", .{ from_version, limit }) catch {
            std.debug.print("Error: Query too long\n", .{});
            return;
        };

    // Parse and execute
    const parsed = tokenizeAndParse(allocator, sql) catch |err| {
        std.debug.print("Parse error: {}\n", .{err});
        return;
    };
    var tokens = parsed.tokens;
    defer tokens.deinit(allocator);

    const stmt_result = exec.executeStatement(&parsed.stmt, &[_]ast.Value{}) catch |err| {
        std.debug.print("Execution error: {}\n", .{err});
        return;
    };

    // Output results
    switch (stmt_result) {
        .diff_result => |diff| {
            var mutable_diff = diff;
            if (opts.json) {
                outputDiffJson(&mutable_diff);
            } else {
                outputDiffTable(&mutable_diff);
            }
        },
        else => {
            std.debug.print("Unexpected result type\n", .{});
        },
    }
}

/// Output version list as JSON
fn outputVersionsJson(versions: []const executor.Executor.VersionInfo) void {
    std.debug.print("[", .{});
    for (versions, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("\n  {{\"version\":{d},\"timestamp\":{d},\"operation\":\"{s}\",\"rowCount\":{d},\"delta\":\"{s}\"}}", .{ v.version, v.timestamp, v.operation, v.row_count, v.delta });
    }
    std.debug.print("\n]\n", .{});
}

/// Output version list as table
fn outputVersionsTable(versions: []const executor.Executor.VersionInfo) void {
    std.debug.print("version | timestamp                 | operation | rowCount | delta\n", .{});
    std.debug.print("--------|---------------------------|-----------|----------|------\n", .{});
    for (versions) |v| {
        std.debug.print("{d:>7} | {s:<25} | {s:<9} | {d:>8} | {s}\n", .{ v.version, v.timestamp_str, v.operation, v.row_count, v.delta });
    }
}

/// Output diff result as JSON
fn outputDiffJson(diff: *executor.Executor.DiffResult) void {
    std.debug.print("{{\"fragments_added\":{d},\"rows_added\":{d},\"fragments_deleted\":{d},\"rows_deleted\":{d},\"added\":", .{
        diff.fragments_added,
        diff.rows_added,
        diff.fragments_deleted,
        diff.rows_deleted,
    });
    output.printResultsJson(&diff.added);
    std.debug.print(",\"deleted\":", .{});
    output.printResultsJson(&diff.deleted);
    std.debug.print("}}\n", .{});
}

/// Output diff result as table
fn outputDiffTable(diff: *executor.Executor.DiffResult) void {
    std.debug.print("=== Diff v{d} → v{d} ===\n", .{ diff.from_version, diff.to_version });
    std.debug.print("Summary: +{d} added, -{d} deleted (from {d} fragments)\n\n", .{ diff.rows_added, diff.rows_deleted, diff.fragments_added + diff.fragments_deleted });

    // Show added rows
    if (diff.added.row_count > 0) {
        std.debug.print("--- Added rows ({d}) ---\n", .{diff.added.row_count});
        output.outputResults(&diff.added, false, false);
        std.debug.print("\n", .{});
    }

    // Show deleted rows
    if (diff.deleted.row_count > 0) {
        std.debug.print("--- Deleted rows ({d}) ---\n", .{diff.deleted.row_count});
        output.outputResults(&diff.deleted, false, false);
    }

    if (diff.added.row_count == 0 and diff.deleted.row_count == 0) {
        std.debug.print("No row-level changes detected.\n", .{});
    }
}

/// Run pipeline from config file
fn runConfigFile(allocator: std.mem.Allocator, config_path: []const u8) !void {
    const content = std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024) catch |err| {
        std.debug.print("Error reading config file '{s}': {}\n", .{ config_path, err });
        return;
    };
    defer allocator.free(content);

    var config = yaml.parse(allocator, content) catch |err| {
        std.debug.print("Error parsing config file: {}\n", .{err});
        return;
    };
    defer config.deinit();

    std.debug.print("Running: {s}\n", .{config.command});

    if (std.mem.eql(u8, config.command, "ingest")) {
        var opts = args.IngestOptions{
            .input = config.input,
            .output = config.output,
        };
        if (config.options.get("format")) |fmt| {
            opts.format = parseFormat(fmt);
        }
        if (config.options.get("delimiter")) |d| {
            if (d.len > 0) opts.delimiter = d[0];
        }
        if (config.options.get("header")) |h| {
            opts.header = !std.mem.eql(u8, h, "false");
        }
        try ingest.run(allocator, opts);
    } else if (std.mem.eql(u8, config.command, "transform")) {
        const opts = args.TransformOptions{
            .input = config.input,
            .output = config.output,
            .select = config.options.get("select"),
            .filter = config.options.get("filter"),
            .rename = config.options.get("rename"),
            .limit = if (config.options.get("limit")) |l| std.fmt.parseInt(usize, l, 10) catch null else null,
        };
        try transform.run(allocator, opts);
    } else if (std.mem.eql(u8, config.command, "enrich")) {
        var opts = args.EnrichOptions{
            .input = config.input,
            .output = config.output,
            .embed = config.options.get("embed"),
        };
        if (config.options.get("model")) |m| {
            if (std.mem.eql(u8, m, "clip")) opts.model = .clip;
        }
        if (config.options.get("index")) |idx| {
            opts.index = idx;
        }
        try enrich.run(allocator, opts);
    } else {
        std.debug.print("Unknown command: {s}\n", .{config.command});
    }
}

fn parseFormat(fmt: []const u8) args.IngestOptions.Format {
    if (std.mem.eql(u8, fmt, "csv")) return .csv;
    if (std.mem.eql(u8, fmt, "tsv")) return .tsv;
    if (std.mem.eql(u8, fmt, "json")) return .json;
    if (std.mem.eql(u8, fmt, "jsonl")) return .jsonl;
    if (std.mem.eql(u8, fmt, "parquet")) return .parquet;
    if (std.mem.eql(u8, fmt, "arrow")) return .arrow;
    if (std.mem.eql(u8, fmt, "avro")) return .avro;
    if (std.mem.eql(u8, fmt, "orc")) return .orc;
    if (std.mem.eql(u8, fmt, "xlsx")) return .xlsx;
    return .auto;
}

/// Find config file in current directory
fn findConfigFile() ?[]const u8 {
    const config_names = [_][]const u8{
        "lanceql.yaml",
        "lanceql.yml",
        ".lanceqlrc.yaml",
    };

    for (config_names) |name| {
        if (std.fs.cwd().access(name, .{})) |_| {
            return name;
        } else |_| {}
    }
    return null;
}

/// Legacy Args struct for backward compatibility with existing query functions
const LegacyArgs = struct {
    query: ?[]const u8 = null,
    file: ?[]const u8 = null,
    benchmark: bool = false,
    iterations: usize = 10,
    warmup: usize = 3,
    json: bool = false,
    help: bool = false,
    show_version: bool = false,
    csv: bool = false,
};

/// Tokenize and parse a SQL query, returning the parsed statement
fn tokenizeAndParse(allocator: std.mem.Allocator, query: []const u8) !struct { stmt: parser.Statement, tokens: std.ArrayList(lexer.Token) } {
    var lex = lexer.Lexer.init(query);
    var tokens = std.ArrayList(lexer.Token){};
    errdefer tokens.deinit(allocator);

    while (true) {
        const tok = try lex.nextToken();
        try tokens.append(allocator, tok);
        if (tok.type == .EOF) break;
    }

    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = try parse.parseStatement();
    return .{ .stmt = stmt, .tokens = tokens };
}

/// Output query results in the format specified by legacy_args
fn outputResults(result: *executor.Result, legacy_args: LegacyArgs) void {
    output.outputResults(result, legacy_args.json, legacy_args.csv);
}

/// Execute query on an already-initialized executor and output results
fn executeAndOutput(allocator: std.mem.Allocator, exec: *executor.Executor, query: []const u8, legacy_args: LegacyArgs) !void {
    const parsed = try tokenizeAndParse(allocator, query);
    var tokens = parsed.tokens;
    defer tokens.deinit(allocator);

    var result = try exec.execute(&parsed.stmt.select, &[_]ast.Value{});
    defer result.deinit();

    outputResults(&result, legacy_args);
}



fn runQuery(allocator: std.mem.Allocator, query: []const u8, legacy_args: LegacyArgs) !void {
    // Extract table path from query
    const table_path = file_utils.extractTablePath(query) orelse {
        std.debug.print("Error: Could not extract table path from query\n", .{});
        std.debug.print("Query should be: SELECT ... FROM 'path/to/file.parquet'\n", .{});
        return;
    };

    // Check for Delta first (directory-based, doesn't need to read file data)
    if (file_detect.isDeltaDirectory(table_path) or std.mem.endsWith(u8, table_path, ".delta")) {
        runQueryWithFormat(allocator, .delta, .{ .path = table_path }, query, legacy_args) catch |err| {
            std.debug.print("Delta query error: {}\n", .{err});
        };
        return;
    }

    // Check for Iceberg (directory-based, doesn't need to read file data)
    if (file_detect.isIcebergDirectory(table_path) or std.mem.endsWith(u8, table_path, ".iceberg")) {
        runQueryWithFormat(allocator, .iceberg, .{ .path = table_path }, query, legacy_args) catch |err| {
            std.debug.print("Iceberg query error: {}\n", .{err});
        };
        return;
    }

    // Read file into memory
    const data = file_utils.openFileOrDataset(allocator, table_path) orelse {
        std.debug.print("Error opening '{s}': file not found or unreadable\n", .{table_path});
        return;
    };
    defer allocator.free(data);

    // Detect file type
    const file_type = file_detect.detect(table_path, data);

    const format: AnyTable.Format = switch (file_type) {
        .parquet => .parquet,
        .lance => .lance,
        .delta => .delta,
        .iceberg => .iceberg,
        .arrow => .arrow,
        .avro => .avro,
        .orc => .orc,
        .xlsx => .xlsx,
        else => .lance, // Try Lance first for unknown/text formats
    };

    const input: AnyTable.InitInput = switch (format) {
        .delta, .iceberg => .{ .path = table_path },
        else => .{ .data = data },
    };

    runQueryWithFormat(allocator, format, input, query, legacy_args) catch |err| {
        if (file_type == .unknown or file_type == .csv or file_type == .json or file_type == .jsonl or file_type == .tsv) {
            // Fallback to Parquet for unknown/text formats
            runQueryWithFormat(allocator, .parquet, .{ .data = data }, query, legacy_args) catch |e| {
                std.debug.print("Query error: {}\n", .{e});
            };
        } else {
            std.debug.print("{s} query error: {}\n", .{ @tagName(format), err });
        }
    };
}

fn runQueryWithFormat(allocator: std.mem.Allocator, format: AnyTable.Format, input: AnyTable.InitInput, query: []const u8, legacy_args: LegacyArgs) !void {
    var table = try AnyTable.init(allocator, format, input);
    defer table.deinit();

    if (format == .iceberg and table.numRows() == 0) {
        std.debug.print("Warning: Iceberg table has no data files\n", .{});
        return;
    }

    var exec = executor.Executor.initWithAnyTable(&table, allocator);
    defer exec.deinit();

    try executeAndOutput(allocator, &exec, query, legacy_args);
}


