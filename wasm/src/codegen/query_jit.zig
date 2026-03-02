//! Query JIT Context - Caching and JIT Compilation for Fused Queries
//!
//! Manages the compilation pipeline for fused query functions:
//! 1. Parses SQL and builds query plan
//! 2. Generates fused Zig code
//! 3. JIT compiles to shared library
//! 4. Caches compiled queries by content hash
//!
//! Usage:
//!   var ctx = QueryJitContext.init(allocator);
//!   defer ctx.deinit();
//!
//!   const result = try ctx.compileQuery(sql, schema);
//!   // result.fn_ptr is the compiled function

const std = @import("std");
const builtin = @import("builtin");
const metal0_jit = @import("metal0_jit.zig");

// SQL modules
const sql_ast = @import("../sql/ast.zig");
const sql_parser = @import("../sql/parser.zig");
const sql_lexer = @import("../sql/lexer.zig");

// Planner and codegen
const planner_mod = @import("../sql/planner/planner.zig");
const plan_nodes = @import("../sql/planner/plan_nodes.zig");
const fused_codegen = @import("../sql/codegen/fused_codegen.zig");

const Planner = planner_mod.Planner;
const QueryPlan = plan_nodes.QueryPlan;
const FusedCodeGen = fused_codegen.FusedCodeGen;
const LanceSchema = metal0_jit.LanceSchema;

/// Compiled query function signature
/// Takes columns, outputs, returns count of rows written
pub const FusedQueryFn = *const fn (
    columns: *const anyopaque,
    output: *anyopaque,
) callconv(.c) usize;

/// Compiled query result
pub const CompiledQuery = struct {
    /// Compiled function pointer
    fn_ptr: ?FusedQueryFn,

    /// Loaded dynamic library
    lib: ?std.DynLib,

    /// Path to .so file
    lib_path: ?[]const u8,

    /// Generated Zig source (for debugging)
    zig_source: []const u8,

    /// Query plan (for inspection)
    plan: ?QueryPlan,

    /// Allocator for cleanup
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CompiledQuery) void {
        if (self.lib) |*lib| {
            lib.close();
        }
        if (self.lib_path) |path| {
            // Optionally delete the temp file
            std.fs.deleteFileAbsolute(path) catch {};
            self.allocator.free(path);
        }
        self.allocator.free(self.zig_source);
    }
};

/// Query JIT Context
pub const QueryJitContext = struct {
    allocator: std.mem.Allocator,

    /// Underlying JIT context for @logic_table compilation
    jit: metal0_jit.JitContext,

    /// Cached compiled queries (hash -> CompiledQuery)
    query_cache: std.AutoHashMap(u64, CompiledQuery),

    /// Statistics
    stats: CacheStats,

    /// Temp directory for generated files
    temp_dir: []const u8,

    const Self = @This();

    pub const CacheStats = struct {
        hits: u64 = 0,
        misses: u64 = 0,
        compile_time_ns: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .jit = metal0_jit.JitContext.init(allocator),
            .query_cache = std.AutoHashMap(u64, CompiledQuery).init(allocator),
            .stats = .{},
            .temp_dir = "/tmp",
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.query_cache.valueIterator();
        while (iter.next()) |compiled| {
            compiled.deinit();
        }
        self.query_cache.deinit();
        self.jit.deinit();
    }

    /// Compile a SQL query to native code
    pub fn compileQuery(
        self: *Self,
        sql: []const u8,
        schema: ?LanceSchema,
    ) !CompiledQuery {
        // 1. Hash query + schema for cache lookup
        const cache_key = computeQueryHash(sql, schema);
        if (self.query_cache.get(cache_key)) |cached| {
            self.stats.hits += 1;
            return cached;
        }
        self.stats.misses += 1;

        const start_time = std.time.nanoTimestamp();

        // 2. Parse SQL to AST
        var lexer = sql_lexer.Lexer.init(sql);
        var parser = sql_parser.Parser.init(self.allocator, &lexer);
        const stmt = try parser.parseSelect();

        // 3. Plan the query
        var query_planner = Planner.init(self.allocator);
        defer query_planner.deinit();
        const plan = try query_planner.plan(&stmt);

        // 4. Generate fused code
        var codegen = FusedCodeGen.init(self.allocator);
        defer codegen.deinit();
        const zig_source = try codegen.generate(&plan);

        // 5. JIT compile
        const compiled = try self.jitCompile(zig_source, cache_key);

        const end_time = std.time.nanoTimestamp();
        self.stats.compile_time_ns += @intCast(@as(i128, end_time - start_time));

        // 6. Cache and return
        try self.query_cache.put(cache_key, compiled);
        return compiled;
    }

    /// Compile Zig source to shared library
    fn jitCompile(self: *Self, zig_source: []const u8, hash: u64) !CompiledQuery {
        // Generate unique temp path
        const lib_ext = switch (builtin.os.tag) {
            .macos => ".dylib",
            .windows => ".dll",
            else => ".so",
        };
        const lib_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/lanceql_query_{x}{s}",
            .{ self.temp_dir, hash, lib_ext },
        );
        errdefer self.allocator.free(lib_path);

        // Write Zig source to temp file
        const zig_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/lanceql_query_{x}.zig",
            .{ self.temp_dir, hash },
        );
        defer self.allocator.free(zig_path);

        const zig_file = try std.fs.createFileAbsolute(zig_path, .{});
        defer zig_file.close();
        try zig_file.writeAll(zig_source);

        // Compile with zig build-lib
        // Explicitly pass env_map to avoid Linux panic when /proc/self/environ is unavailable
        var env_map = std.process.EnvMap.init(self.allocator);
        defer env_map.deinit();
        const env_vars = [_][]const u8{ "PATH", "HOME", "USER", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "ZIG_LOCAL_CACHE_DIR", "ZIG_GLOBAL_CACHE_DIR" };
        for (env_vars) |key| {
            if (std.posix.getenv(key)) |value| {
                env_map.put(key, value) catch {};
            }
        }

        const result = try std.process.Child.run(.{
            .allocator = self.allocator,
            .argv = &.{
                "zig",
                "build-lib",
                "-dynamic",
                "-O", "ReleaseFast",
                "-femit-bin=" ++ lib_path,
                zig_path,
            },
            .env_map = &env_map,
        });
        defer self.allocator.free(result.stdout);
        defer self.allocator.free(result.stderr);

        if (result.term.Exited != 0) {
            std.log.err("Zig compilation failed: {s}", .{result.stderr});
            return error.CompilationFailed;
        }

        // Load the compiled library
        var lib = std.DynLib.open(lib_path) catch |err| {
            std.log.err("Failed to load compiled library: {}", .{err});
            return error.LoadFailed;
        };

        // Lookup the function
        const fn_ptr = lib.lookup(FusedQueryFn, "fused_query");

        return CompiledQuery{
            .fn_ptr = fn_ptr,
            .lib = lib,
            .lib_path = lib_path,
            .zig_source = try self.allocator.dupe(u8, zig_source),
            .plan = null,
            .allocator = self.allocator,
        };
    }

    /// Compute cache key from query and schema
    fn computeQueryHash(sql: []const u8, schema: ?LanceSchema) u64 {
        var hasher = std.hash.Fnv1a_64.init();
        hasher.update(sql);

        if (schema) |s| {
            for (s.columns) |col| {
                hasher.update(col.name);
                hasher.update(&.{@intFromEnum(col.column_type)});
            }
        }

        return hasher.final();
    }

    /// Get cache statistics
    pub fn getStats(self: *const Self) CacheStats {
        return self.stats;
    }

    /// Clear the query cache
    pub fn clearCache(self: *Self) void {
        var iter = self.query_cache.valueIterator();
        while (iter.next()) |compiled| {
            compiled.deinit();
        }
        self.query_cache.clearRetainingCapacity();
        self.stats = .{};
    }
};

// ============================================================================
// Tests
// ============================================================================

test "QueryJitContext init/deinit" {
    const allocator = std.testing.allocator;
    var ctx = QueryJitContext.init(allocator);
    defer ctx.deinit();
}

test "computeQueryHash" {
    const hash1 = QueryJitContext.computeQueryHash("SELECT * FROM t", null);
    const hash2 = QueryJitContext.computeQueryHash("SELECT * FROM t", null);
    const hash3 = QueryJitContext.computeQueryHash("SELECT * FROM t2", null);

    try std.testing.expectEqual(hash1, hash2);
    try std.testing.expect(hash1 != hash3);
}
