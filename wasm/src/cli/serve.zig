//! LanceQL Serve Command
//!
//! HTTP server for SQL queries and vector search.
//!
//! Usage:
//!   lanceql serve data.lance --port 3000
//!   lanceql serve data.parquet --host 0.0.0.0 --no-open

const std = @import("std");
const args = @import("args.zig");
const http = @import("http.zig");
const ingest = @import("ingest.zig");
const transform = @import("transform.zig");
const enrich = @import("enrich.zig");

// SQL execution
const lanceql = @import("lanceql");
const Table = lanceql.Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const ArrowTable = @import("lanceql.arrow_table").ArrowTable;

const lexer = @import("lanceql.sql.lexer");
const parser = @import("lanceql.sql.parser");
const executor = @import("lanceql.sql.executor");
const ast = @import("lanceql.sql.ast");

pub const ServeError = error{
    BindFailed,
    AcceptFailed,
    ReadFailed,
    WriteFailed,
    FileNotFound,
    InvalidFormat,
    QueryError,
};

/// Embedded HTML UI
const INDEX_HTML = @embedFile("serve_ui.html");
const CONFIG_HTML = @embedFile("config_ui.html");

/// File types for detection
const FileType = enum {
    lance,
    parquet,
    arrow,
    unknown,
};

/// Serve mode
const ServeMode = enum {
    single_file,
    folder,
    config,
};

/// Server state
pub const Server = struct {
    allocator: std.mem.Allocator,
    listener: std.net.Server,
    data_path: ?[]const u8,
    file_data: ?[]const u8,
    host: []const u8,
    port: u16,
    mode: ServeMode,
    folder_path: ?[]const u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, opts: args.ServeOptions) !Self {
        const address = std.net.Address.parseIp4(opts.host, opts.port) catch |err| {
            std.debug.print("Error parsing address {s}:{}: {}\n", .{ opts.host, opts.port, err });
            return ServeError.BindFailed;
        };

        const listener = std.net.Address.listen(address, .{
            .reuse_address = true,
        }) catch |err| {
            std.debug.print("Error binding to {s}:{}: {}\n", .{ opts.host, opts.port, err });
            return ServeError.BindFailed;
        };

        // Detect folder vs file mode
        var mode: ServeMode = .single_file;
        var folder_path: ?[]const u8 = null;
        var file_data: ?[]const u8 = null;
        var data_path: ?[]const u8 = opts.input;

        if (opts.input) |path| {
            const stat = std.fs.cwd().statFile(path) catch |err| {
                std.debug.print("Error accessing '{s}': {}\n", .{ path, err });
                return ServeError.FileNotFound;
            };

            if (stat.kind == .directory) {
                mode = .folder;
                folder_path = path;
                data_path = null;
            } else {
                file_data = std.fs.cwd().readFileAlloc(allocator, path, 500 * 1024 * 1024) catch |err| {
                    std.debug.print("Error reading file '{s}': {}\n", .{ path, err });
                    return ServeError.FileNotFound;
                };
            }
        }

        return Self{
            .allocator = allocator,
            .listener = listener,
            .data_path = data_path,
            .file_data = file_data,
            .host = opts.host,
            .port = opts.port,
            .mode = mode,
            .folder_path = folder_path,
        };
    }

    pub fn deinit(self: *Self) void {
        self.listener.deinit();
        if (self.file_data) |data| {
            self.allocator.free(data);
        }
    }

    pub fn getUrl(self: *const Self) []const u8 {
        return std.fmt.allocPrint(self.allocator, "http://{s}:{}", .{ self.host, self.port }) catch "http://localhost:3000";
    }

    /// Run the server loop
    pub fn run(self: *Self) !void {
        std.debug.print("\nLanceQL Server running at http://{s}:{}\n", .{ self.host, self.port });
        switch (self.mode) {
            .folder => {
                std.debug.print("Mode: folder browser\n", .{});
                if (self.folder_path) |path| {
                    std.debug.print("Folder: {s}\n", .{path});
                }
            },
            .single_file => {
                if (self.data_path) |path| {
                    std.debug.print("Serving: {s}\n", .{path});
                }
            },
            .config => {
                std.debug.print("Mode: config builder\n", .{});
            },
        }
        std.debug.print("Press Ctrl+C to stop\n\n", .{});

        while (true) {
            const connection = self.listener.accept() catch |err| {
                std.debug.print("Accept error: {}\n", .{err});
                continue;
            };

            self.handleConnection(connection) catch |err| {
                std.debug.print("Connection error: {}\n", .{err});
            };
        }
    }

    fn handleConnection(self: *Self, connection: std.net.Server.Connection) !void {
        defer connection.stream.close();

        // Read request (up to 64KB)
        var buffer: [65536]u8 = undefined;
        const bytes_read = connection.stream.read(&buffer) catch |err| {
            std.debug.print("Read error: {}\n", .{err});
            return;
        };

        if (bytes_read == 0) return;

        // Parse HTTP request
        var request = http.parseRequest(self.allocator, buffer[0..bytes_read]) catch {
            const resp = try http.errorResponse(self.allocator, 400, "Invalid request");
            var resp_copy = resp;
            defer resp_copy.deinit();
            const response_bytes = try resp_copy.toBytes();
            defer self.allocator.free(response_bytes);
            _ = connection.stream.write(response_bytes) catch {};
            return;
        };
        defer request.deinit();

        // Log request
        std.debug.print("{s} {s}\n", .{ @tagName(request.method), request.path });

        // Route request
        var response = self.route(&request) catch |err| {
            std.debug.print("Route error: {}\n", .{err});
            var err_resp = http.errorResponse(self.allocator, 500, "Internal server error") catch return;
            defer err_resp.deinit();
            const err_bytes = err_resp.toBytes() catch return;
            defer self.allocator.free(err_bytes);
            _ = connection.stream.write(err_bytes) catch {};
            return;
        };
        defer response.deinit();

        // Send response
        const response_bytes = try response.toBytes();
        defer self.allocator.free(response_bytes);
        _ = connection.stream.write(response_bytes) catch {};
    }

    fn route(self: *Self, request: *http.Request) !http.Response {
        if (request.method == .OPTIONS) {
            return http.corsPreflightResponse(self.allocator);
        }

        if (std.mem.eql(u8, request.path, "/")) {
            return self.handleIndex();
        } else if (std.mem.eql(u8, request.path, "/api/query")) {
            return self.handleQuery(request);
        } else if (std.mem.eql(u8, request.path, "/api/schema")) {
            return self.handleSchema();
        } else if (std.mem.eql(u8, request.path, "/api/files")) {
            return self.handleFiles();
        } else if (std.mem.eql(u8, request.path, "/api/health")) {
            return http.jsonResponse(self.allocator, "{\"status\":\"ok\"}");
        } else if (std.mem.eql(u8, request.path, "/api/run")) {
            return self.handleRun(request);
        } else if (std.mem.startsWith(u8, request.path, "/api/schema/")) {
            const file_path = request.path["/api/schema/".len..];
            return self.handleSchemaForFile(file_path);
        } else {
            return http.errorResponse(self.allocator, 404, "Not found");
        }
    }

    fn handleFiles(self: *Self) !http.Response {
        if (self.mode != .folder or self.folder_path == null) {
            return http.jsonResponse(self.allocator, "{\"files\":[],\"mode\":\"single_file\"}");
        }

        const folder = self.folder_path.?;
        var json = std.ArrayListUnmanaged(u8){};
        defer json.deinit(self.allocator);

        try json.appendSlice(self.allocator, "{\"files\":[");

        var dir = std.fs.cwd().openDir(folder, .{ .iterate = true }) catch {
            return http.errorResponse(self.allocator, 500, "Cannot open folder");
        };
        defer dir.close();

        var first = true;
        var iter = dir.iterate();
        while (iter.next() catch null) |entry| {
            const is_data_file = isDataFile(entry.name);
            if (!is_data_file and entry.kind != .directory) continue;

            if (!first) try json.appendSlice(self.allocator, ",");
            first = false;

            const kind_str = if (entry.kind == .directory) "folder" else "file";
            const file_type = if (entry.kind == .directory)
                detectFolderType(self.allocator, folder, entry.name)
            else
                getFileTypeFromName(entry.name);

            try json.writer(self.allocator).print("{{\"name\":\"{s}\",\"kind\":\"{s}\",\"type\":\"{s}\"}}", .{
                entry.name,
                kind_str,
                file_type,
            });
        }

        try json.appendSlice(self.allocator, "],\"mode\":\"folder\"}");
        const json_str = try self.allocator.dupe(u8, json.items);
        return http.jsonResponse(self.allocator, json_str);
    }

    fn handleSchemaForFile(self: *Self, file_path: []const u8) !http.Response {
        if (self.folder_path == null) {
            return http.errorResponse(self.allocator, 400, "Not in folder mode");
        }

        const full_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.folder_path.?, file_path });
        defer self.allocator.free(full_path);

        const data = std.fs.cwd().readFileAlloc(self.allocator, full_path, 500 * 1024 * 1024) catch {
            return http.errorResponse(self.allocator, 404, "File not found");
        };
        defer self.allocator.free(data);

        const file_type = detectFileType(file_path, data);
        return self.buildSchemaResponse(file_path, data, file_type);
    }

    fn buildSchemaResponse(self: *Self, path: ?[]const u8, data: []const u8, file_type: FileType) !http.Response {
        var json = std.ArrayListUnmanaged(u8){};
        defer json.deinit(self.allocator);

        try json.appendSlice(self.allocator, "{\"columns\":[");

        var row_count: usize = 0;
        switch (file_type) {
            .parquet => {
                var table = ParquetTable.init(self.allocator, data) catch {
                    return http.errorResponse(self.allocator, 500, "Failed to parse file");
                };
                defer table.deinit();
                row_count = table.numRows();
                for (table.getColumnNames(), 0..) |col_name, i| {
                    if (i > 0) try json.appendSlice(self.allocator, ",");
                    const col_type = if (table.getColumnType(i)) |t| @tagName(t) else "unknown";
                    try json.writer(self.allocator).print("{{\"name\":\"{s}\",\"type\":\"{s}\"}}", .{ col_name, col_type });
                }
            },
            .lance => {
                var table = Table.init(self.allocator, data) catch {
                    return http.errorResponse(self.allocator, 500, "Failed to parse file");
                };
                defer table.deinit();
                if (table.schema) |schema| {
                    for (schema.fields, 0..) |field, i| {
                        if (i > 0) try json.appendSlice(self.allocator, ",");
                        try json.writer(self.allocator).print("{{\"name\":\"{s}\",\"type\":\"{s}\"}}", .{ field.name, field.logical_type });
                    }
                }
            },
            .arrow => {
                var table = ArrowTable.init(self.allocator, data) catch {
                    return http.errorResponse(self.allocator, 500, "Failed to parse file");
                };
                defer table.deinit();
                row_count = table.numRows();
                for (table.getColumnNames(), 0..) |col_name, i| {
                    if (i > 0) try json.appendSlice(self.allocator, ",");
                    const col_type = if (table.getColumnType(i)) |t| @tagName(t) else "unknown";
                    try json.writer(self.allocator).print("{{\"name\":\"{s}\",\"type\":\"{s}\"}}", .{ col_name, col_type });
                }
            },
            .unknown => {
                return http.errorResponse(self.allocator, 400, "Unknown file format");
            },
        }

        if (path) |p| {
            try json.writer(self.allocator).print("],\"row_count\":{},\"path\":\"{s}\"}}", .{ row_count, p });
        } else {
            try json.writer(self.allocator).print("],\"row_count\":{}}}", .{row_count});
        }

        const json_str = try self.allocator.dupe(u8, json.items);
        return http.jsonResponse(self.allocator, json_str);
    }

    fn handleIndex(self: *Self) !http.Response {
        if (self.mode == .config) {
            return http.htmlResponse(self.allocator, CONFIG_HTML);
        }
        return http.htmlResponse(self.allocator, INDEX_HTML);
    }

    fn handleRun(self: *Self, request: *http.Request) !http.Response {
        if (request.method != .POST) {
            return http.errorResponse(self.allocator, 405, "Method not allowed");
        }

        // Parse command from JSON body
        const command = extractJsonField(request.body, "command") orelse {
            return http.errorResponse(self.allocator, 400, "Missing 'command' field");
        };

        const input_field = extractJsonField(request.body, "input");
        const output_field = extractJsonField(request.body, "output");

        if (std.mem.eql(u8, command, "ingest")) {
            var opts = args.IngestOptions{
                .input = input_field,
                .output = output_field,
            };
            if (extractJsonField(request.body, "format")) |fmt| {
                opts.format = parseFormat(fmt);
            }
            ingest.run(self.allocator, opts) catch |err| {
                const msg = std.fmt.allocPrint(self.allocator, "{}", .{err}) catch "execution failed";
                return http.errorResponse(self.allocator, 500, msg);
            };
            return http.jsonResponse(self.allocator, "{\"success\":true}");
        } else if (std.mem.eql(u8, command, "transform")) {
            const opts = args.TransformOptions{
                .input = input_field,
                .output = output_field,
                .select = extractJsonField(request.body, "select"),
                .filter = extractJsonField(request.body, "filter"),
            };
            transform.run(self.allocator, opts) catch |err| {
                const msg = std.fmt.allocPrint(self.allocator, "{}", .{err}) catch "execution failed";
                return http.errorResponse(self.allocator, 500, msg);
            };
            return http.jsonResponse(self.allocator, "{\"success\":true}");
        } else if (std.mem.eql(u8, command, "enrich")) {
            var opts = args.EnrichOptions{
                .input = input_field,
                .output = output_field,
                .embed = extractJsonField(request.body, "embed"),
            };
            if (extractJsonField(request.body, "model")) |m| {
                if (std.mem.eql(u8, m, "clip")) opts.model = .clip;
            }
            enrich.run(self.allocator, opts) catch |err| {
                const msg = std.fmt.allocPrint(self.allocator, "{}", .{err}) catch "execution failed";
                return http.errorResponse(self.allocator, 500, msg);
            };
            return http.jsonResponse(self.allocator, "{\"success\":true}");
        }

        return http.errorResponse(self.allocator, 400, "Unknown command");
    }

    fn handleSchema(self: *Self) !http.Response {
        if (self.data_path == null or self.file_data == null) {
            return http.jsonResponse(self.allocator, "{\"columns\":[],\"row_count\":0}");
        }

        const path = self.data_path.?;
        const data = self.file_data.?;
        const file_type = detectFileType(path, data);

        return self.buildSchemaResponse(null, data, file_type);
    }

    fn handleQuery(self: *Self, request: *http.Request) !http.Response {
        if (request.method != .POST) {
            return http.errorResponse(self.allocator, 405, "Method not allowed");
        }

        // Parse JSON body to extract SQL
        const sql = extractSqlFromJson(request.body) orelse {
            return http.errorResponse(self.allocator, 400, "Missing 'sql' field in request body");
        };

        // Build full query - inject FROM clause if needed
        const full_sql = try self.buildFullQuery(sql);
        defer self.allocator.free(full_sql);

        // Execute query
        const result_json = self.executeQuery(full_sql) catch |err| {
            const msg = try std.fmt.allocPrint(self.allocator, "Query error: {}", .{err});
            defer self.allocator.free(msg);
            return http.errorResponse(self.allocator, 400, msg);
        };

        return http.jsonResponse(self.allocator, result_json);
    }

    /// Build complete SQL with FROM clause if missing
    fn buildFullQuery(self: *Self, sql: []const u8) ![]const u8 {
        const path = self.data_path orelse return try self.allocator.dupe(u8, sql);

        // Check if query already has FROM clause
        const upper_sql = try self.allocator.dupe(u8, sql);
        defer self.allocator.free(upper_sql);
        for (upper_sql, 0..) |c, i| {
            upper_sql[i] = std.ascii.toUpper(c);
        }

        if (std.mem.indexOf(u8, upper_sql, "FROM") != null) {
            // Already has FROM, use as-is
            return try self.allocator.dupe(u8, sql);
        }

        // Find position to insert FROM clause (after SELECT columns, before LIMIT/WHERE/ORDER/GROUP)
        const keywords = [_][]const u8{ "LIMIT", "WHERE", "ORDER", "GROUP", "HAVING" };
        var insert_pos: usize = sql.len;

        for (keywords) |kw| {
            if (std.mem.indexOf(u8, upper_sql, kw)) |pos| {
                if (pos < insert_pos) insert_pos = pos;
            }
        }

        // Build: SELECT ... FROM 'path' [rest]
        return try std.fmt.allocPrint(self.allocator, "{s} FROM '{s}' {s}", .{
            std.mem.trimRight(u8, sql[0..insert_pos], " \t\n\r"),
            path,
            sql[insert_pos..],
        });
    }

    fn executeQuery(self: *Self, sql: []const u8) ![]const u8 {
        if (self.data_path == null or self.file_data == null) {
            return error.QueryError;
        }

        const data = self.file_data.?;
        const path = self.data_path.?;
        const file_type = detectFileType(path, data);

        // Tokenize
        var lex = lexer.Lexer.init(sql);
        var tokens = std.ArrayList(lexer.Token){};
        defer tokens.deinit(self.allocator);

        while (true) {
            const tok = try lex.nextToken();
            try tokens.append(self.allocator, tok);
            if (tok.type == .EOF) break;
        }

        // Parse
        var parse = parser.Parser.init(tokens.items, self.allocator);
        const stmt = try parse.parseStatement();

        // Execute based on file type
        var result = switch (file_type) {
            .parquet => blk: {
                var table = try ParquetTable.init(self.allocator, data);
                defer table.deinit();
                var exec = executor.Executor.initWithParquet(&table, self.allocator);
                defer exec.deinit();
                break :blk try exec.execute(&stmt.select, &[_]ast.Value{});
            },
            .lance => blk: {
                var table = try Table.init(self.allocator, data);
                defer table.deinit();
                var exec = executor.Executor.init(&table, self.allocator);
                defer exec.deinit();
                break :blk try exec.execute(&stmt.select, &[_]ast.Value{});
            },
            .arrow => blk: {
                var table = try ArrowTable.init(self.allocator, data);
                defer table.deinit();
                var exec = executor.Executor.initWithArrow(&table, self.allocator);
                defer exec.deinit();
                break :blk try exec.execute(&stmt.select, &[_]ast.Value{});
            },
            .unknown => return error.InvalidFormat,
        };
        defer result.deinit();

        // Convert result to JSON
        return try resultToJson(self.allocator, &result);
    }
};

/// Detect file type from extension and magic bytes
fn detectFileType(path: []const u8, data: []const u8) FileType {
    if (std.mem.endsWith(u8, path, ".lance")) return .lance;
    if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
    if (std.mem.endsWith(u8, path, ".arrow") or std.mem.endsWith(u8, path, ".feather")) return .arrow;

    if (data.len >= 4 and std.mem.eql(u8, data[0..4], "PAR1")) return .parquet;
    if (data.len >= 6 and std.mem.eql(u8, data[0..6], "ARROW1")) return .arrow;
    if (data.len >= 40 and std.mem.eql(u8, data[data.len - 4 ..], "LANC")) return .lance;

    return .unknown;
}

/// Check if filename is a supported data file
fn isDataFile(name: []const u8) bool {
    const extensions = [_][]const u8{ ".lance", ".parquet", ".arrow", ".feather", ".csv", ".json", ".jsonl" };
    for (extensions) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}

/// Get file type string from filename
fn getFileTypeFromName(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ".lance")) return "lance";
    if (std.mem.endsWith(u8, name, ".parquet")) return "parquet";
    if (std.mem.endsWith(u8, name, ".arrow") or std.mem.endsWith(u8, name, ".feather")) return "arrow";
    if (std.mem.endsWith(u8, name, ".csv")) return "csv";
    if (std.mem.endsWith(u8, name, ".json") or std.mem.endsWith(u8, name, ".jsonl")) return "json";
    return "unknown";
}

/// Detect folder type (Delta Lake, Iceberg, or Lance)
fn detectFolderType(allocator: std.mem.Allocator, parent: []const u8, name: []const u8) []const u8 {
    const full_path = std.fmt.allocPrint(allocator, "{s}/{s}", .{ parent, name }) catch return "folder";
    defer allocator.free(full_path);

    var dir = std.fs.cwd().openDir(full_path, .{ .iterate = true }) catch return "folder";
    defer dir.close();

    var iter = dir.iterate();
    while (iter.next() catch null) |entry| {
        if (std.mem.eql(u8, entry.name, "_delta_log")) return "delta";
        if (std.mem.eql(u8, entry.name, "metadata")) return "iceberg";
        if (std.mem.endsWith(u8, entry.name, ".lance")) return "lance";
    }
    return "folder";
}

/// Extract SQL from JSON body: {"sql": "..."}
fn extractSqlFromJson(body: []const u8) ?[]const u8 {
    return extractJsonField(body, "sql");
}

/// Extract a string field from JSON body
fn extractJsonField(body: []const u8, field: []const u8) ?[]const u8 {
    // Build key pattern: "field"
    var key_buf: [64]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "\"{s}\"", .{field}) catch return null;

    const start = std.mem.indexOf(u8, body, key) orelse return null;
    const after_key = body[start + key.len ..];

    const colon = std.mem.indexOf(u8, after_key, ":") orelse return null;
    const after_colon = std.mem.trimLeft(u8, after_key[colon + 1 ..], " \t\n\r");

    if (after_colon.len == 0 or after_colon[0] != '"') return null;
    const value_start = after_colon[1..];

    var i: usize = 0;
    while (i < value_start.len) : (i += 1) {
        if (value_start[i] == '\\' and i + 1 < value_start.len) {
            i += 1;
        } else if (value_start[i] == '"') {
            return value_start[0..i];
        }
    }
    return null;
}

/// Parse format string to IngestOptions.Format
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

/// Convert executor result to JSON
fn resultToJson(allocator: std.mem.Allocator, result: *executor.Result) ![]const u8 {
    var json = std.ArrayListUnmanaged(u8){};
    errdefer json.deinit(allocator);

    // Start object
    try json.appendSlice(allocator, "{\"columns\":[");

    // Column names
    for (result.columns, 0..) |col, i| {
        if (i > 0) try json.appendSlice(allocator, ",");
        try json.writer(allocator).print("\"{s}\"", .{col.name});
    }

    try json.appendSlice(allocator, "],\"rows\":[");

    // Rows
    const row_count = result.row_count;
    for (0..row_count) |row| {
        if (row > 0) try json.appendSlice(allocator, ",");
        try json.appendSlice(allocator, "[");

        for (result.columns, 0..) |col, col_idx| {
            if (col_idx > 0) try json.appendSlice(allocator, ",");

            switch (col.data) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| {
                    try json.writer(allocator).print("{}", .{data[row]});
                },
                .int32, .date32 => |data| {
                    try json.writer(allocator).print("{}", .{data[row]});
                },
                .float64 => |data| {
                    try json.writer(allocator).print("{d:.6}", .{data[row]});
                },
                .float32 => |data| {
                    try json.writer(allocator).print("{d:.6}", .{data[row]});
                },
                .bool_ => |data| {
                    try json.appendSlice(allocator, if (data[row]) "true" else "false");
                },
                .string => |data| {
                    try json.appendSlice(allocator, "\"");
                    // Escape special characters
                    for (data[row]) |c| {
                        switch (c) {
                            '"' => try json.appendSlice(allocator, "\\\""),
                            '\\' => try json.appendSlice(allocator, "\\\\"),
                            '\n' => try json.appendSlice(allocator, "\\n"),
                            '\r' => try json.appendSlice(allocator, "\\r"),
                            '\t' => try json.appendSlice(allocator, "\\t"),
                            else => try json.append(allocator, c),
                        }
                    }
                    try json.appendSlice(allocator, "\"");
                },
            }
        }

        try json.appendSlice(allocator, "]");
    }

    try json.writer(allocator).print("],\"row_count\":{}}}", .{row_count});

    return json.toOwnedSlice(allocator);
}

/// Open browser to URL (platform-specific)
pub fn openBrowser(allocator: std.mem.Allocator, url: []const u8) void {
    const cmd = switch (@import("builtin").os.tag) {
        .macos => &[_][]const u8{ "open", url },
        .linux => &[_][]const u8{ "xdg-open", url },
        .windows => &[_][]const u8{ "cmd", "/c", "start", url },
        else => return,
    };

    // Explicitly build env_map to avoid Linux panic when /proc/self/environ is unavailable
    var env_map = std.process.EnvMap.init(allocator);
    defer env_map.deinit();
    const env_vars = [_][]const u8{ "PATH", "HOME", "USER", "DISPLAY", "XDG_RUNTIME_DIR" };
    for (env_vars) |key| {
        if (std.posix.getenv(key)) |value| {
            env_map.put(key, value) catch {};
        }
    }

    var child = std.process.Child.init(cmd, allocator);
    child.env_map = &env_map;
    child.spawn() catch return;
}

/// Run the serve command
pub fn run(allocator: std.mem.Allocator, opts: args.ServeOptions) !void {
    var server = try Server.init(allocator, opts);
    defer server.deinit();

    if (opts.open) {
        const url = try std.fmt.allocPrint(allocator, "http://{s}:{}", .{ opts.host, opts.port });
        defer allocator.free(url);
        openBrowser(allocator, url);
    }

    try server.run();
}

/// Run config builder mode - opens browser with config UI
pub fn runConfigMode(allocator: std.mem.Allocator, command: []const u8) !void {
    const address = std.net.Address.parseIp4("127.0.0.1", 0) catch return;
    const listener = std.net.Address.listen(address, .{ .reuse_address = true }) catch return;

    const port = listener.listen_address.getPort();
    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{}", .{port});
    defer allocator.free(url);

    std.debug.print("\nLanceQL Config Builder\n", .{});
    std.debug.print("Command: {s}\n", .{command});
    std.debug.print("Opening: {s}\n", .{url});
    std.debug.print("Press Ctrl+C to cancel\n\n", .{});

    openBrowser(allocator, url);

    var server = Server{
        .allocator = allocator,
        .listener = listener,
        .data_path = null,
        .file_data = null,
        .host = "127.0.0.1",
        .port = port,
        .mode = .config,
        .folder_path = null,
    };
    defer server.deinit();

    try server.run();
}

// =============================================================================
// Tests
// =============================================================================

test "extract sql from json" {
    const sql1 = extractSqlFromJson("{\"sql\": \"SELECT * FROM data\"}");
    try std.testing.expectEqualStrings("SELECT * FROM data", sql1.?);

    const sql2 = extractSqlFromJson("{\"sql\":\"SELECT 1\"}");
    try std.testing.expectEqualStrings("SELECT 1", sql2.?);

    const sql3 = extractSqlFromJson("{}");
    try std.testing.expect(sql3 == null);
}

test "detect file type" {
    try std.testing.expectEqual(FileType.parquet, detectFileType("test.parquet", "PAR1...."));
    try std.testing.expectEqual(FileType.lance, detectFileType("test.lance", "...."));
    try std.testing.expectEqual(FileType.arrow, detectFileType("test.arrow", "...."));
}
