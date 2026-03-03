//! QueryMode CLI Help Messages
//!
//! All command help output functions.

const std = @import("std");

/// Print main help
pub fn printHelp() void {
    std.debug.print(
        \\QueryMode - High-performance data pipeline for Lance files
        \\
        \\Usage: querymode [command] [options]
        \\
        \\Commands:
        \\  query, q      Execute SQL query on Lance/Parquet files
        \\  ingest, i     Convert CSV/JSON/Parquet to Lance format
        \\  transform, t  Transform Lance data (select, filter, rename)
        \\  enrich, e     Add embeddings and indexes to Lance data
        \\  serve, s      Start interactive web server with WebGPU UI
        \\  history, h    Show version history for a Lance table
        \\  diff, d       Show differences between versions
        \\  help          Show this help message
        \\  version       Show version
        \\
        \\Global Options:
        \\  -c, --config <file>  Use config file (YAML)
        \\  -v, --verbose        Enable verbose output
        \\  -h, --help           Show help
        \\  -V, --version        Show version
        \\
        \\Examples:
        \\  querymode query "SELECT * FROM 'data.lance' LIMIT 10"
        \\  querymode ingest data.csv -o dataset.lance
        \\  querymode enrich dataset.lance --embed text --model minilm
        \\  querymode serve dataset.lance
        \\  querymode                    # Auto-detect: config or serve
        \\
        \\Run 'querymode <command> --help' for command-specific help.
        \\
    , .{});
}

/// Print query command help
pub fn printQueryHelp() void {
    std.debug.print(
        \\Usage: querymode query [options] "SQL QUERY"
        \\
        \\Execute SQL queries on Lance and Parquet files.
        \\
        \\Options:
        \\  -c, --command <SQL>    Execute SQL query
        \\  -f, --file <PATH>      Read SQL from file
        \\  -b, --benchmark        Run query in benchmark mode
        \\  -i, --iterations <N>   Benchmark iterations (default: 10)
        \\  -w, --warmup <N>       Warmup iterations (default: 3)
        \\      --json             Output results as JSON
        \\      --csv              Output results as CSV
        \\  -h, --help             Show this help
        \\
        \\Examples:
        \\  querymode query "SELECT * FROM 'users.lance' WHERE age > 25"
        \\  querymode query -b "SELECT COUNT(*) FROM 'data.parquet'"
        \\  querymode query -f query.sql --json
        \\
    , .{});
}

/// Print ingest command help
pub fn printIngestHelp() void {
    std.debug.print(
        \\Usage: querymode ingest <input> -o <output.lance> [options]
        \\
        \\Convert data files to Lance format.
        \\
        \\Supported formats (auto-detected from extension):
        \\  csv, tsv          Delimiter-separated values
        \\  json, jsonl       JSON array or newline-delimited JSON
        \\  parquet           Apache Parquet columnar format
        \\  arrow             Apache Arrow IPC/Feather (.arrow, .arrows, .feather)
        \\  avro              Apache Avro container format
        \\  orc               Apache ORC columnar format
        \\  xlsx              Microsoft Excel (uncompressed)
        \\  delta             Delta Lake table (directory)
        \\  iceberg           Apache Iceberg table (directory)
        \\
        \\Options:
        \\  -o, --output <PATH>    Output Lance file (required)
        \\      --format <FMT>     Override auto-detected format
        \\      --glob <PATTERN>   Glob pattern for directory input (e.g., "*.csv")
        \\  -d, --delimiter <C>    CSV delimiter character (default: auto-detect)
        \\      --no-header        CSV has no header row
        \\      --schema <FILE>    Schema file (JSON)
        \\  -h, --help             Show this help
        \\
        \\Examples:
        \\  querymode ingest data.csv -o dataset.lance
        \\  querymode ingest data.json --format jsonl -o dataset.lance
        \\  querymode ingest data.parquet -o dataset.lance
        \\  querymode ingest data.arrow -o dataset.lance
        \\  querymode ingest ./my_delta_table/ --format delta -o dataset.lance
        \\
    , .{});
}

/// Print serve command help
pub fn printServeHelp() void {
    std.debug.print(
        \\Usage: querymode serve [input] [options]
        \\
        \\Start interactive web server with WebGPU-powered UI.
        \\
        \\Options:
        \\  -p, --port <N>         Server port (default: 3000)
        \\      --host <ADDR>      Host address (default: 127.0.0.1)
        \\      --no-open          Don't auto-open browser
        \\  -h, --help             Show this help
        \\
        \\Features:
        \\  - Infinite scroll table with WebGPU acceleration
        \\  - SQL editor with syntax highlighting
        \\  - Vector search with embedding preview
        \\  - Auto-embedding generation
        \\  - Timeline/version navigation
        \\
        \\Examples:
        \\  querymode serve dataset.lance
        \\  querymode serve ./datasets/ --port 8080
        \\  querymode serve                          # Serve current directory
        \\
    , .{});
}

pub fn printTransformHelp() void {
    std.debug.print(
        \\Usage: querymode transform <input> -o <output> [options]
        \\
        \\Apply transformations to Lance data.
        \\Use 'querymode query' with SQL for transformations.
        \\
        \\Options:
        \\  -o, --output <PATH>    Output file (required)
        \\      --select <COLS>    Columns to select
        \\      --filter <EXPR>    Filter expression
        \\  -h, --help             Show this help
        \\
    , .{});
}

pub fn printEnrichHelp() void {
    std.debug.print(
        \\Usage: querymode enrich <input> -o <output> [options]
        \\
        \\Add embeddings and vector indexes to Lance data.
        \\
        \\Options:
        \\      --embed <COLUMN>   Column to embed
        \\      --model <NAME>     Embedding model (minilm, clip)
        \\      --index <COLUMN>   Create vector index on column
        \\      --index-type <T>   Index type (ivf-pq, flat)
        \\  -o, --output <PATH>    Output file (required)
        \\  -h, --help             Show this help
        \\
        \\Examples:
        \\  querymode enrich data.lance --embed text -o enriched.lance
        \\  querymode enrich data.parquet --embed desc --model clip -o out.lance
        \\
    , .{});
}

/// Print history command help
pub fn printHistoryHelp() void {
    std.debug.print(
        \\Usage: querymode history <table.lance> [options]
        \\
        \\Show version history for a Lance table.
        \\
        \\Options:
        \\  -l, --limit <N>        Limit to last N versions
        \\      --json             Output as JSON
        \\  -h, --help             Show this help
        \\
        \\Output columns:
        \\  version    Version number
        \\  timestamp  When the version was created
        \\  operation  Type of operation (INSERT, DELETE, UPDATE)
        \\  rowCount   Number of rows at this version
        \\  delta      Change in row count (+N or -N)
        \\
        \\Examples:
        \\  querymode history users.lance
        \\  querymode history users.lance --limit 5
        \\  querymode history users.lance --json
        \\
    , .{});
}

/// Print diff command help
pub fn printDiffHelp() void {
    std.debug.print(
        \\Usage: querymode diff <table.lance> --from <N> [--to <M>] [options]
        \\
        \\Show differences between two versions of a Lance table.
        \\
        \\Options:
        \\  -f, --from <N>         Source version (required, or -N for relative)
        \\  -t, --to <M>           Target version (default: HEAD/current)
        \\  -l, --limit <N>        Limit output rows (default: 100)
        \\      --json             Output as JSON
        \\  -h, --help             Show this help
        \\
        \\Version syntax:
        \\  N                      Absolute version number
        \\  -N                     Relative: N versions ago (e.g., -1 = previous)
        \\
        \\Output columns:
        \\  change     Type of change (ADD or DELETE)
        \\  ...        All columns from the table
        \\
        \\Examples:
        \\  querymode diff users.lance --from 2 --to 3
        \\  querymode diff users.lance --from -1          # What changed last?
        \\  querymode diff users.lance -f 1               # Changes since version 1
        \\  querymode diff users.lance -f 2 --limit 1000
        \\
    , .{});
}
