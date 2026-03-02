//! Shared query execution utilities for CLI commands
//!
//! Provides common SQL query execution logic used across transform, enrich, and other CLI commands.

const std = @import("std");
const lanceql = @import("lanceql");
const Table = lanceql.Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const ArrowTable = @import("lanceql.arrow_table").ArrowTable;
const AvroTable = @import("lanceql.avro_table").AvroTable;
const OrcTable = @import("lanceql.orc_table").OrcTable;
const XlsxTable = @import("lanceql.xlsx_table").XlsxTable;
const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const executor = @import("lanceql.sql.executor");
const ast = @import("lanceql.sql.ast");
const file_detect = @import("file_detect.zig");

pub const Result = executor.Result;

pub const QueryError = error{
    LexerError,
    ParseError,
    ExecutionError,
    UnsupportedFormat,
    OutOfMemory,
};

/// Execute SQL query on in-memory data
pub fn executeQuery(
    allocator: std.mem.Allocator,
    data: []const u8,
    sql: []const u8,
    file_type: file_detect.FileType,
) (QueryError || error{InvalidStatement})!Result {
    // Tokenize
    var lex = lexer.Lexer.init(sql);
    var tokens = std.ArrayList(lexer.Token){};
    defer tokens.deinit(allocator);

    while (true) {
        const tok = lex.nextToken() catch return QueryError.LexerError;
        tokens.append(allocator, tok) catch return QueryError.OutOfMemory;
        if (tok.type == .EOF) break;
    }

    // Parse
    var parse = parser.Parser.init(tokens.items, allocator);
    const stmt = parse.parseStatement() catch return QueryError.ParseError;

    // Execute based on file type
    return switch (file_type) {
        .parquet => executeWithParquet(allocator, data, &stmt),
        .lance => executeWithLance(allocator, data, &stmt),
        .arrow => executeWithArrow(allocator, data, &stmt),
        .avro => executeWithAvro(allocator, data, &stmt),
        .orc => executeWithOrc(allocator, data, &stmt),
        .xlsx => executeWithXlsx(allocator, data, &stmt),
        else => executeWithFallback(allocator, data, &stmt),
    };
}

fn executeWithParquet(allocator: std.mem.Allocator, data: []const u8, stmt: *const parser.Statement) !Result {
    var pq_table = ParquetTable.init(allocator, data) catch return QueryError.ExecutionError;
    defer pq_table.deinit();

    var exec = executor.Executor.initWithParquet(&pq_table, allocator);
    defer exec.deinit();

    return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
}

fn executeWithLance(allocator: std.mem.Allocator, data: []const u8, stmt: *const parser.Statement) !Result {
    var table = Table.init(allocator, data) catch return QueryError.ExecutionError;
    defer table.deinit();

    var exec = executor.Executor.init(&table, allocator);
    defer exec.deinit();

    return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
}

fn executeWithArrow(allocator: std.mem.Allocator, data: []const u8, stmt: *const parser.Statement) !Result {
    var arrow_table = ArrowTable.init(allocator, data) catch return QueryError.ExecutionError;
    defer arrow_table.deinit();

    var exec = executor.Executor.initWithArrow(&arrow_table, allocator);
    defer exec.deinit();

    return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
}

fn executeWithAvro(allocator: std.mem.Allocator, data: []const u8, stmt: *const parser.Statement) !Result {
    var avro_table = AvroTable.init(allocator, data) catch return QueryError.ExecutionError;
    defer avro_table.deinit();

    var exec = executor.Executor.initWithAvro(&avro_table, allocator);
    defer exec.deinit();

    return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
}

fn executeWithOrc(allocator: std.mem.Allocator, data: []const u8, stmt: *const parser.Statement) !Result {
    var orc_table = OrcTable.init(allocator, data) catch return QueryError.ExecutionError;
    defer orc_table.deinit();

    var exec = executor.Executor.initWithOrc(&orc_table, allocator);
    defer exec.deinit();

    return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
}

fn executeWithXlsx(allocator: std.mem.Allocator, data: []const u8, stmt: *const parser.Statement) !Result {
    var xlsx_table = XlsxTable.init(allocator, data) catch return QueryError.ExecutionError;
    defer xlsx_table.deinit();

    var exec = executor.Executor.initWithXlsx(&xlsx_table, allocator);
    defer exec.deinit();

    return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
}

fn executeWithFallback(allocator: std.mem.Allocator, data: []const u8, stmt: *const parser.Statement) !Result {
    // Try Lance first, then Parquet as fallback
    if (Table.init(allocator, data)) |table_result| {
        var table = table_result;
        defer table.deinit();
        var exec = executor.Executor.init(&table, allocator);
        defer exec.deinit();
        return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
    } else |_| {
        var pq_table = ParquetTable.init(allocator, data) catch return QueryError.ExecutionError;
        defer pq_table.deinit();
        var exec = executor.Executor.initWithParquet(&pq_table, allocator);
        defer exec.deinit();
        return exec.execute(&stmt.select, &[_]ast.Value{}) catch return QueryError.ExecutionError;
    }
}
