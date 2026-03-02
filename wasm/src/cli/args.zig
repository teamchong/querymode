//! LanceQL CLI Argument Parser
//!
//! Supports subcommand-based architecture:
//!   lanceql query "SELECT ..."
//!   lanceql ingest data.csv -o out.lance
//!   lanceql transform data.lance --select "a,b"
//!   lanceql enrich data.lance --embed text
//!   lanceql serve data.lance
//!   lanceql (no args) - auto-detect config or serve

const std = @import("std");
const help = @import("help.zig");

pub const version = "0.2.0";

/// Subcommands
pub const Command = enum {
    query,
    ingest,
    transform,
    enrich,
    serve,
    history,
    diff,
    help,
    version,
    none, // No command specified - auto-detect mode
};

/// Global options (apply to all commands)
pub const GlobalOptions = struct {
    help: bool = false,
    show_version: bool = false,
    verbose: bool = false,
    config: ?[]const u8 = null, // -c, --config
};

/// Query command options
pub const QueryOptions = struct {
    query: ?[]const u8 = null,
    file: ?[]const u8 = null,
    benchmark: bool = false,
    iterations: usize = 10,
    warmup: usize = 3,
    json: bool = false,
    csv: bool = false,
    help: bool = false,
};

/// Ingest command options
pub const IngestOptions = struct {
    input: ?[]const u8 = null, // Positional: input file/dir
    output: ?[]const u8 = null, // -o, --output
    format: Format = .auto, // --format
    glob: ?[]const u8 = null, // --glob pattern
    delimiter: ?u8 = null, // --delimiter for CSV
    header: bool = true, // --no-header to disable
    schema: ?[]const u8 = null, // --schema file
    help: bool = false,

    pub const Format = enum {
        auto,
        csv,
        tsv,
        json,
        jsonl,
        parquet,
        arrow,
        avro,
        orc,
        xlsx,
        delta,
        iceberg,
    };
};

/// Transform command options
pub const TransformOptions = struct {
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    select: ?[]const u8 = null, // --select "col1,col2"
    filter: ?[]const u8 = null, // --filter "x > 100"
    rename: ?[]const u8 = null, // --rename "old:new"
    cast: ?[]const u8 = null, // --cast "col:type"
    limit: ?usize = null, // --limit N
    help: bool = false,
};

/// Enrich command options
pub const EnrichOptions = struct {
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    embed: ?[]const u8 = null, // --embed column
    model: Model = .minilm, // --model
    index: ?[]const u8 = null, // --index column
    index_type: IndexType = .ivf_pq, // --index-type
    partitions: usize = 256, // --partitions
    help: bool = false,

    pub const Model = enum {
        minilm,
        clip,
    };

    pub const IndexType = enum {
        ivf_pq,
        flat,
    };
};

/// Serve command options
pub const ServeOptions = struct {
    input: ?[]const u8 = null, // Positional: file/dir to serve
    port: u16 = 3000, // --port
    host: []const u8 = "127.0.0.1", // --host
    open: bool = true, // --no-open to disable
    help: bool = false,
};

/// History command options
pub const HistoryOptions = struct {
    input: ?[]const u8 = null, // Positional: Lance file/table
    limit: ?usize = null, // --limit N
    json: bool = false, // --json output
    help: bool = false,
};

/// Diff command options
pub const DiffOptions = struct {
    input: ?[]const u8 = null, // Positional: Lance file/table
    from: ?i64 = null, // --from N (version number, or -N for relative)
    to: ?i64 = null, // --to N (version number, defaults to HEAD)
    limit: ?usize = null, // --limit N (default 100)
    json: bool = false, // --json output
    help: bool = false,
};

/// Parsed arguments
pub const Args = struct {
    command: Command,
    global: GlobalOptions,
    query: QueryOptions,
    ingest: IngestOptions,
    transform: TransformOptions,
    enrich: EnrichOptions,
    serve: ServeOptions,
    history: HistoryOptions,
    diff_opts: DiffOptions,
    remaining: []const []const u8, // Unparsed args
};

pub fn parse(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);

    var args = Args{
        .command = .none,
        .global = .{},
        .query = .{},
        .ingest = .{},
        .transform = .{},
        .enrich = .{},
        .serve = .{},
        .history = .{},
        .diff_opts = .{},
        .remaining = &[_][]const u8{},
    };

    if (argv.len < 2) {
        return args; // No arguments = auto-detect mode
    }

    var i: usize = 1;

    // Check for global flags first
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.global.help = true;
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            args.global.show_version = true;
            args.command = .version;
            return args;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            args.global.verbose = true;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--config")) {
            i += 1;
            if (i < argv.len) args.global.config = argv[i];
        } else {
            break; // Not a global flag, check for command
        }
    }

    if (i >= argv.len) {
        if (args.global.help) args.command = .help;
        return args;
    }

    // Parse command
    const cmd_str = argv[i];
    if (std.mem.eql(u8, cmd_str, "query") or std.mem.eql(u8, cmd_str, "q")) {
        args.command = .query;
        i += 1;
        try parseQueryOptions(&args.query, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "ingest") or std.mem.eql(u8, cmd_str, "i")) {
        args.command = .ingest;
        i += 1;
        try parseIngestOptions(&args.ingest, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "transform") or std.mem.eql(u8, cmd_str, "t")) {
        args.command = .transform;
        i += 1;
        try parseTransformOptions(&args.transform, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "enrich") or std.mem.eql(u8, cmd_str, "e")) {
        args.command = .enrich;
        i += 1;
        try parseEnrichOptions(&args.enrich, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "serve") or std.mem.eql(u8, cmd_str, "s")) {
        args.command = .serve;
        i += 1;
        try parseServeOptions(&args.serve, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "history") or std.mem.eql(u8, cmd_str, "h")) {
        args.command = .history;
        i += 1;
        try parseHistoryOptions(&args.history, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "diff") or std.mem.eql(u8, cmd_str, "d")) {
        args.command = .diff;
        i += 1;
        try parseDiffOptions(&args.diff_opts, argv, &i);
    } else if (std.mem.eql(u8, cmd_str, "help")) {
        args.command = .help;
    } else if (std.mem.eql(u8, cmd_str, "version")) {
        args.command = .version;
    } else if (!std.mem.startsWith(u8, cmd_str, "-")) {
        // No recognized command - treat as query (backward compat)
        args.command = .query;
        args.query.query = cmd_str;
        i += 1;
        try parseQueryOptions(&args.query, argv, &i);
    } else {
        // Flags without command - parse as query options for backward compat
        args.command = .query;
        try parseQueryOptions(&args.query, argv, &i);
    }

    return args;
}

fn parseQueryOptions(opts: *QueryOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--command")) {
            i.* += 1;
            if (i.* < argv.len) opts.query = argv[i.*];
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--file")) {
            i.* += 1;
            if (i.* < argv.len) opts.file = argv[i.*];
        } else if (std.mem.eql(u8, arg, "-b") or std.mem.eql(u8, arg, "--benchmark")) {
            opts.benchmark = true;
        } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--iterations")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.iterations = std.fmt.parseInt(usize, argv[i.*], 10) catch 10;
            }
        } else if (std.mem.eql(u8, arg, "-w") or std.mem.eql(u8, arg, "--warmup")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.warmup = std.fmt.parseInt(usize, argv[i.*], 10) catch 3;
            }
        } else if (std.mem.eql(u8, arg, "--json")) {
            opts.json = true;
        } else if (std.mem.eql(u8, arg, "--csv")) {
            opts.csv = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            // Positional = query string
            if (opts.query == null) opts.query = arg;
        }
    }
}

fn parseIngestOptions(opts: *IngestOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i.* += 1;
            if (i.* < argv.len) opts.output = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--format")) {
            i.* += 1;
            if (i.* < argv.len) {
                const fmt = argv[i.*];
                if (std.mem.eql(u8, fmt, "csv")) opts.format = .csv
                else if (std.mem.eql(u8, fmt, "tsv")) opts.format = .tsv
                else if (std.mem.eql(u8, fmt, "json")) opts.format = .json
                else if (std.mem.eql(u8, fmt, "jsonl")) opts.format = .jsonl
                else if (std.mem.eql(u8, fmt, "parquet")) opts.format = .parquet
                else if (std.mem.eql(u8, fmt, "arrow")) opts.format = .arrow
                else if (std.mem.eql(u8, fmt, "avro")) opts.format = .avro
                else if (std.mem.eql(u8, fmt, "orc")) opts.format = .orc
                else if (std.mem.eql(u8, fmt, "xlsx") or std.mem.eql(u8, fmt, "excel")) opts.format = .xlsx
                else if (std.mem.eql(u8, fmt, "delta")) opts.format = .delta
                else if (std.mem.eql(u8, fmt, "iceberg")) opts.format = .iceberg;
            }
        } else if (std.mem.eql(u8, arg, "--glob")) {
            i.* += 1;
            if (i.* < argv.len) opts.glob = argv[i.*];
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--delimiter")) {
            i.* += 1;
            if (i.* < argv.len and argv[i.*].len > 0) opts.delimiter = argv[i.*][0];
        } else if (std.mem.eql(u8, arg, "--no-header")) {
            opts.header = false;
        } else if (std.mem.eql(u8, arg, "--schema")) {
            i.* += 1;
            if (i.* < argv.len) opts.schema = argv[i.*];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            // Positional = input file
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseTransformOptions(opts: *TransformOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i.* += 1;
            if (i.* < argv.len) opts.output = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--select")) {
            i.* += 1;
            if (i.* < argv.len) opts.select = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--filter")) {
            i.* += 1;
            if (i.* < argv.len) opts.filter = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--rename")) {
            i.* += 1;
            if (i.* < argv.len) opts.rename = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--cast")) {
            i.* += 1;
            if (i.* < argv.len) opts.cast = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--limit")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.limit = std.fmt.parseInt(usize, argv[i.*], 10) catch null;
            }
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseEnrichOptions(opts: *EnrichOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i.* += 1;
            if (i.* < argv.len) opts.output = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--embed")) {
            i.* += 1;
            if (i.* < argv.len) opts.embed = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--model")) {
            i.* += 1;
            if (i.* < argv.len) {
                const m = argv[i.*];
                if (std.mem.eql(u8, m, "minilm")) opts.model = .minilm
                else if (std.mem.eql(u8, m, "clip")) opts.model = .clip;
            }
        } else if (std.mem.eql(u8, arg, "--index")) {
            i.* += 1;
            if (i.* < argv.len) opts.index = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--index-type")) {
            i.* += 1;
            if (i.* < argv.len) {
                const t = argv[i.*];
                if (std.mem.eql(u8, t, "ivf-pq")) opts.index_type = .ivf_pq
                else if (std.mem.eql(u8, t, "flat")) opts.index_type = .flat;
            }
        } else if (std.mem.eql(u8, arg, "--partitions")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.partitions = std.fmt.parseInt(usize, argv[i.*], 10) catch 256;
            }
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseServeOptions(opts: *ServeOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--port")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.port = std.fmt.parseInt(u16, argv[i.*], 10) catch 3000;
            }
        } else if (std.mem.eql(u8, arg, "--host")) {
            i.* += 1;
            if (i.* < argv.len) opts.host = argv[i.*];
        } else if (std.mem.eql(u8, arg, "--no-open")) {
            opts.open = false;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseHistoryOptions(opts: *HistoryOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "-l") or std.mem.eql(u8, arg, "--limit")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.limit = std.fmt.parseInt(usize, argv[i.*], 10) catch null;
            }
        } else if (std.mem.eql(u8, arg, "--json")) {
            opts.json = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

fn parseDiffOptions(opts: *DiffOptions, argv: []const []const u8, i: *usize) !void {
    while (i.* < argv.len) : (i.* += 1) {
        const arg = argv[i.*];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        } else if (std.mem.eql(u8, arg, "--from") or std.mem.eql(u8, arg, "-f")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.from = std.fmt.parseInt(i64, argv[i.*], 10) catch null;
            }
        } else if (std.mem.eql(u8, arg, "--to") or std.mem.eql(u8, arg, "-t")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.to = std.fmt.parseInt(i64, argv[i.*], 10) catch null;
            }
        } else if (std.mem.eql(u8, arg, "-l") or std.mem.eql(u8, arg, "--limit")) {
            i.* += 1;
            if (i.* < argv.len) {
                opts.limit = std.fmt.parseInt(usize, argv[i.*], 10) catch null;
            }
        } else if (std.mem.eql(u8, arg, "--json")) {
            opts.json = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.input == null) opts.input = arg;
        }
    }
}

// Re-export help functions for backward compatibility
pub const printHelp = help.printHelp;
pub const printQueryHelp = help.printQueryHelp;
pub const printIngestHelp = help.printIngestHelp;
pub const printServeHelp = help.printServeHelp;
pub const printTransformHelp = help.printTransformHelp;
pub const printEnrichHelp = help.printEnrichHelp;
pub const printHistoryHelp = help.printHistoryHelp;
pub const printDiffHelp = help.printDiffHelp;

// ============================================================================
// Tests
// ============================================================================

test "args: QueryOptions defaults" {
    const opts = QueryOptions{};
    try std.testing.expectEqual(@as(?[]const u8, null), opts.query);
    try std.testing.expectEqual(@as(?[]const u8, null), opts.file);
    try std.testing.expect(!opts.benchmark);
    try std.testing.expectEqual(@as(usize, 10), opts.iterations);
    try std.testing.expectEqual(@as(usize, 3), opts.warmup);
    try std.testing.expect(!opts.json);
    try std.testing.expect(!opts.csv);
}

test "args: parseQueryOptions with positional query" {
    var opts = QueryOptions{};
    const argv = [_][]const u8{ "SELECT * FROM t", "--json" };
    var i: usize = 0;
    try parseQueryOptions(&opts, &argv, &i);

    try std.testing.expectEqualStrings("SELECT * FROM t", opts.query.?);
    try std.testing.expect(opts.json);
    try std.testing.expect(!opts.csv);
}

test "args: parseQueryOptions with benchmark flags" {
    var opts = QueryOptions{};
    const argv = [_][]const u8{ "-b", "-i", "20", "-w", "5", "SELECT 1" };
    var i: usize = 0;
    try parseQueryOptions(&opts, &argv, &i);

    try std.testing.expect(opts.benchmark);
    try std.testing.expectEqual(@as(usize, 20), opts.iterations);
    try std.testing.expectEqual(@as(usize, 5), opts.warmup);
    try std.testing.expectEqualStrings("SELECT 1", opts.query.?);
}

test "args: parseQueryOptions with file flag" {
    var opts = QueryOptions{};
    const argv = [_][]const u8{ "-f", "query.sql", "--csv" };
    var i: usize = 0;
    try parseQueryOptions(&opts, &argv, &i);

    try std.testing.expectEqualStrings("query.sql", opts.file.?);
    try std.testing.expect(opts.csv);
}

test "args: IngestOptions defaults" {
    const opts = IngestOptions{};
    try std.testing.expectEqual(@as(?[]const u8, null), opts.input);
    try std.testing.expectEqual(@as(?[]const u8, null), opts.output);
    try std.testing.expectEqual(IngestOptions.Format.auto, opts.format);
    try std.testing.expect(opts.header);
}

test "args: parseIngestOptions with format" {
    var opts = IngestOptions{};
    const argv = [_][]const u8{ "data.csv", "-o", "out.lance", "--format", "csv" };
    var i: usize = 0;
    try parseIngestOptions(&opts, &argv, &i);

    try std.testing.expectEqualStrings("data.csv", opts.input.?);
    try std.testing.expectEqualStrings("out.lance", opts.output.?);
    try std.testing.expectEqual(IngestOptions.Format.csv, opts.format);
}

test "args: parseIngestOptions format variants" {
    const formats = [_]struct { str: []const u8, expected: IngestOptions.Format }{
        .{ .str = "csv", .expected = .csv },
        .{ .str = "tsv", .expected = .tsv },
        .{ .str = "json", .expected = .json },
        .{ .str = "jsonl", .expected = .jsonl },
        .{ .str = "parquet", .expected = .parquet },
        .{ .str = "arrow", .expected = .arrow },
        .{ .str = "avro", .expected = .avro },
        .{ .str = "orc", .expected = .orc },
        .{ .str = "xlsx", .expected = .xlsx },
        .{ .str = "excel", .expected = .xlsx },
        .{ .str = "delta", .expected = .delta },
        .{ .str = "iceberg", .expected = .iceberg },
    };

    for (formats) |f| {
        var opts = IngestOptions{};
        const argv = [_][]const u8{ "--format", f.str };
        var i: usize = 0;
        try parseIngestOptions(&opts, &argv, &i);
        try std.testing.expectEqual(f.expected, opts.format);
    }
}

test "args: parseIngestOptions with delimiter and no-header" {
    var opts = IngestOptions{};
    const argv = [_][]const u8{ "data.txt", "-d", "|", "--no-header" };
    var i: usize = 0;
    try parseIngestOptions(&opts, &argv, &i);

    try std.testing.expectEqual(@as(?u8, '|'), opts.delimiter);
    try std.testing.expect(!opts.header);
}

test "args: parseServeOptions with port" {
    var opts = ServeOptions{};
    const argv = [_][]const u8{ "dataset.lance", "-p", "8080", "--no-open" };
    var i: usize = 0;
    try parseServeOptions(&opts, &argv, &i);

    try std.testing.expectEqualStrings("dataset.lance", opts.input.?);
    try std.testing.expectEqual(@as(u16, 8080), opts.port);
    try std.testing.expect(!opts.open);
}

test "args: parseServeOptions with host" {
    var opts = ServeOptions{};
    const argv = [_][]const u8{ "--host", "0.0.0.0", "-p", "3001" };
    var i: usize = 0;
    try parseServeOptions(&opts, &argv, &i);

    try std.testing.expectEqualStrings("0.0.0.0", opts.host);
    try std.testing.expectEqual(@as(u16, 3001), opts.port);
}

test "args: parseTransformOptions" {
    var opts = TransformOptions{};
    const argv = [_][]const u8{ "input.lance", "-o", "output.lance", "--select", "a,b,c", "--filter", "x > 10", "--limit", "100" };
    var i: usize = 0;
    try parseTransformOptions(&opts, &argv, &i);

    try std.testing.expectEqualStrings("input.lance", opts.input.?);
    try std.testing.expectEqualStrings("output.lance", opts.output.?);
    try std.testing.expectEqualStrings("a,b,c", opts.select.?);
    try std.testing.expectEqualStrings("x > 10", opts.filter.?);
    try std.testing.expectEqual(@as(?usize, 100), opts.limit);
}

test "args: parseEnrichOptions with model and index" {
    var opts = EnrichOptions{};
    const argv = [_][]const u8{ "data.lance", "--embed", "text", "--model", "clip", "--index-type", "flat", "--partitions", "128" };
    var i: usize = 0;
    try parseEnrichOptions(&opts, &argv, &i);

    try std.testing.expectEqualStrings("data.lance", opts.input.?);
    try std.testing.expectEqualStrings("text", opts.embed.?);
    try std.testing.expectEqual(EnrichOptions.Model.clip, opts.model);
    try std.testing.expectEqual(EnrichOptions.IndexType.flat, opts.index_type);
    try std.testing.expectEqual(@as(usize, 128), opts.partitions);
}

test "args: GlobalOptions defaults" {
    const opts = GlobalOptions{};
    try std.testing.expect(!opts.help);
    try std.testing.expect(!opts.show_version);
    try std.testing.expect(!opts.verbose);
    try std.testing.expectEqual(@as(?[]const u8, null), opts.config);
}
