//! EdgeQ CLI Help Messages
//!
//! All command help output functions.

const std = @import("std");

/// Print main help
pub fn printHelp() void {
    std.debug.print(
        \\EdgeQ - High-performance data pipeline for Lance files
        \\
        \\Usage: edgeq [command] [options]
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
        \\  edgeq query "SELECT * FROM 'data.lance' LIMIT 10"
        \\  edgeq ingest data.csv -o dataset.lance
        \\  edgeq enrich dataset.lance --embed text --model minilm
        \\  edgeq serve dataset.lance
        \\  edgeq                    # Auto-detect: config or serve
        \\
        \\Run 'edgeq <command> --help' for command-specific help.
        \\
    , .{});
}

/// Print query command help
pub fn printQueryHelp() void {
    std.debug.print(
        \\Usage: edgeq query [options] "SQL QUERY"
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
        \\  edgeq query "SELECT * FROM 'users.lance' WHERE age > 25"
        \\  edgeq query -b "SELECT COUNT(*) FROM 'data.parquet'"
        \\  edgeq query -f query.sql --json
        \\
    , .{});
}

/// Print ingest command help
pub fn printIngestHelp() void {
    std.debug.print(
        \\Usage: edgeq ingest <input> -o <output.lance> [options]
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
        \\  edgeq ingest data.csv -o dataset.lance
        \\  edgeq ingest data.json --format jsonl -o dataset.lance
        \\  edgeq ingest data.parquet -o dataset.lance
        \\  edgeq ingest data.arrow -o dataset.lance
        \\  edgeq ingest ./my_delta_table/ --format delta -o dataset.lance
        \\
    , .{});
}

/// Print serve command help
pub fn printServeHelp() void {
    std.debug.print(
        \\Usage: edgeq serve [input] [options]
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
        \\  edgeq serve dataset.lance
        \\  edgeq serve ./datasets/ --port 8080
        \\  edgeq serve                          # Serve current directory
        \\
    , .{});
}

pub fn printTransformHelp() void {
    std.debug.print(
        \\Usage: edgeq transform <input> -o <output> [options]
        \\
        \\Apply transformations to Lance data.
        \\Use 'edgeq query' with SQL for transformations.
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
        \\Usage: edgeq enrich <input> -o <output> [options]
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
        \\  edgeq enrich data.lance --embed text -o enriched.lance
        \\  edgeq enrich data.parquet --embed desc --model clip -o out.lance
        \\
    , .{});
}

/// Print history command help
pub fn printHistoryHelp() void {
    std.debug.print(
        \\Usage: edgeq history <table.lance> [options]
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
        \\  edgeq history users.lance
        \\  edgeq history users.lance --limit 5
        \\  edgeq history users.lance --json
        \\
    , .{});
}

/// Print diff command help
pub fn printDiffHelp() void {
    std.debug.print(
        \\Usage: edgeq diff <table.lance> --from <N> [--to <M>] [options]
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
        \\  edgeq diff users.lance --from 2 --to 3
        \\  edgeq diff users.lance --from -1          # What changed last?
        \\  edgeq diff users.lance -f 1               # Changes since version 1
        \\  edgeq diff users.lance -f 2 --limit 1000
        \\
    , .{});
}
