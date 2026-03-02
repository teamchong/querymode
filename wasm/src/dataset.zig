//! LanceDataset - High-level API for Lance datasets with time travel support.
//!
//! Mirrors the browser vault.js API for consistency across CLI and browser.
//! Both use SQL execution internally, sharing the same executor code.
//!
//! Example:
//!   const dataset = try LanceDataset.open(allocator, "data.lance");
//!   defer dataset.deinit();
//!
//!   // Time travel (uses SQL: SHOW VERSIONS, DIFF)
//!   const versions = try dataset.timeline();
//!   const diff = try dataset.diff(.{ .from = 1, .to = 2 });
//!   const last = try dataset.lastChange();
//!
//!   // Query with DataFrame
//!   const df = try dataset.df();
//!   const result = try df.select(&.{"id", "name"}).limit(10).collect();

const std = @import("std");
const executor = @import("lanceql.sql.executor");
const lexer = @import("lexer");
const parser = @import("parser");
const ast = @import("ast");
const Table = @import("lanceql.table").Table;
const DataFrame = @import("lanceql.dataframe").DataFrame;
const format = @import("lanceql.format");

/// Version information returned by timeline().
/// Matches browser vault.js structure and executor.VersionInfo.
pub const VersionInfo = executor.Executor.VersionInfo;

/// Diff result returned by diff().
/// Matches browser vault.js structure and executor.DiffResult.
pub const DiffResult = executor.Executor.DiffResult;

/// Diff options matching browser vault.js API.
pub const DiffOptions = struct {
    from: i32,
    to: ?u32 = null,
    limit: u32 = 100,
};

/// LanceDataset - High-level API for Lance datasets.
/// Mirrors browser vault.js for API consistency.
/// Uses SQL execution internally, sharing code with browser WASM.
pub const LanceDataset = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    exec: executor.Executor,

    const Self = @This();

    /// Open a Lance dataset from a path.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Self {
        // Verify path exists and is a Lance dataset
        const versions_path = try std.fs.path.join(allocator, &.{ path, "_versions" });
        defer allocator.free(versions_path);

        std.fs.cwd().access(versions_path, .{}) catch {
            return error.DatasetNotFound;
        };

        // Initialize executor with dataset path
        var exec = executor.Executor.init(null, allocator);
        exec.setDatasetPath(path);

        return Self{
            .allocator = allocator,
            .path = try allocator.dupe(u8, path),
            .exec = exec,
        };
    }

    /// Clean up resources.
    pub fn deinit(self: *Self) void {
        self.exec.deinit();
        self.allocator.free(self.path);
    }

    // =========================================================================
    // Time Travel APIs (mirrors browser vault.js)
    // Uses SQL execution - same code path as browser WASM
    // =========================================================================

    /// Get version timeline for this dataset.
    /// Executes: SHOW VERSIONS FOR <path>
    ///
    /// Returns version history with deltas, matching browser vault.js structure.
    pub fn timeline(self: *Self) ![]const VersionInfo {
        return self.timelineWithLimit(null);
    }

    /// Get version timeline with limit.
    /// Executes: SHOW VERSIONS FOR <path> LIMIT n
    pub fn timelineWithLimit(self: *Self, limit: ?u32) ![]const VersionInfo {
        // Build SQL: SHOW VERSIONS FOR 'path' [LIMIT n]
        var sql_buf: [512]u8 = undefined;
        const sql = if (limit) |l|
            try std.fmt.bufPrint(&sql_buf, "SHOW VERSIONS FOR read_lance('{s}') LIMIT {d}", .{ self.path, l })
        else
            try std.fmt.bufPrint(&sql_buf, "SHOW VERSIONS FOR read_lance('{s}')", .{self.path});

        // Parse and execute
        const stmt = try self.parseStatement(sql);
        const result = try self.exec.executeStatement(&stmt, &[_]ast.Value{});

        return switch (result) {
            .versions_list => |versions| versions,
            else => error.UnexpectedResult,
        };
    }

    /// Get diff between two versions.
    /// Executes: DIFF <path> VERSION from AND VERSION to LIMIT limit
    ///
    /// Matches browser vault.js diff() API.
    pub fn diff(self: *Self, options: DiffOptions) !DiffResult {
        const from_version: u32 = if (options.from < 0) blk: {
            // Negative means relative to HEAD (e.g., -1 = previous version)
            const latest = try format.manifest.latestVersion(self.allocator, self.path);
            const offset: u32 = @intCast(-options.from);
            break :blk if (latest > offset) latest - offset else 1;
        } else @intCast(options.from);

        const to_version: u32 = options.to orelse try format.manifest.latestVersion(self.allocator, self.path);

        // Build SQL: DIFF 'path' VERSION from AND VERSION to LIMIT limit
        var sql_buf: [512]u8 = undefined;
        const sql = try std.fmt.bufPrint(&sql_buf, "DIFF read_lance('{s}') VERSION {d} AND VERSION {d} LIMIT {d}", .{ self.path, from_version, to_version, options.limit });

        // Parse and execute
        const stmt = try self.parseStatement(sql);
        const result = try self.exec.executeStatement(&stmt, &[_]ast.Value{});

        return switch (result) {
            .diff_result => |d| d,
            else => error.UnexpectedResult,
        };
    }

    /// Get what changed in the last version.
    /// Shorthand for diff with from=-1.
    /// Matches browser vault.js lastChange() API.
    pub fn lastChange(self: *Self) !DiffResult {
        return self.diff(.{ .from = -1 });
    }

    // =========================================================================
    // Query APIs
    // =========================================================================

    /// Execute a SQL query on this dataset.
    /// Equivalent to browser vault.exec(sql).
    pub fn exec_sql(self: *Self, sql: []const u8) !executor.Result {
        const stmt = try self.parseStatement(sql);
        return switch (try self.exec.executeStatement(&stmt, &[_]ast.Value{})) {
            .result => |r| r,
            else => error.UnexpectedResult,
        };
    }

    /// Create a DataFrame for querying this dataset.
    /// Loads the latest version of the data.
    pub fn df(self: *Self) !DataFrame {
        const latest = try format.manifest.latestVersion(self.allocator, self.path);
        return self.dfAtVersion(latest);
    }

    /// Create a DataFrame for querying at a specific version.
    pub fn dfAtVersion(self: *Self, version: u32) !DataFrame {
        var manifest = try format.manifest.loadManifest(self.allocator, self.path, version);
        defer manifest.deinit();

        if (manifest.fragments.len == 0) {
            return error.NoFragments;
        }

        const frag_path = try std.fs.path.join(self.allocator, &.{ self.path, "data", manifest.fragments[0].file_path });
        defer self.allocator.free(frag_path);

        const file_data = try std.fs.cwd().readFileAlloc(self.allocator, frag_path, 100 * 1024 * 1024);

        const table = try self.allocator.create(Table);
        table.* = try Table.init(self.allocator, file_data);

        return DataFrame.from(self.allocator, table);
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    fn parseStatement(self: *Self, sql: []const u8) !parser.Statement {
        var lex = lexer.Lexer.init(sql);
        var tokens: std.ArrayListUnmanaged(lexer.Token) = .empty;
        defer tokens.deinit(self.allocator);

        while (true) {
            const tok = try lex.nextToken();
            try tokens.append(self.allocator, tok);
            if (tok.type == .EOF) break;
        }

        var parse = parser.Parser.init(tokens.items, self.allocator);
        return try parse.parseStatement();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "LanceDataset compiles" {
    // Just verify the types compile correctly
    _ = LanceDataset;
    _ = VersionInfo;
    _ = DiffResult;
    _ = DiffOptions;
}
