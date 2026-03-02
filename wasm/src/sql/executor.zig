//! SQL Executor - Execute parsed SQL queries against Lance and Parquet files
//!
//! Takes a parsed AST and executes it against Lance or Parquet columnar files,
//! returning results in columnar format compatible with better-sqlite3.

const std = @import("std");
const ast = @import("ast");
const Table = @import("lanceql.table").Table;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const format = @import("lanceql.format");
const meta = format.parquet_metadata;
const DeltaTable = @import("lanceql.delta_table").DeltaTable;
const IcebergTable = @import("lanceql.iceberg_table").IcebergTable;
const ArrowTable = @import("lanceql.arrow_table").ArrowTable;
const AvroTable = @import("lanceql.avro_table").AvroTable;
const OrcTable = @import("lanceql.orc_table").OrcTable;
const XlsxTable = @import("lanceql.xlsx_table").XlsxTable;
const AnyTable = @import("lanceql.any_table").AnyTable;
const hash = @import("lanceql.hash");
const vector_engine = @import("lanceql.vector_engine");
const gpu_hash_join = @import("lanceql.query").gpu_hash_join;
pub const logic_table_dispatch = @import("logic_table_dispatch.zig");
pub const scalar_functions = @import("scalar_functions.zig");
pub const aggregate_functions = @import("aggregate_functions.zig");
pub const window_functions = @import("window_functions.zig");
pub const result_types = @import("result_types.zig");
pub const result_ops = @import("result_ops.zig");
pub const having_eval = @import("having_eval.zig");
pub const set_ops = @import("set_ops.zig");
pub const window_eval = @import("window_eval.zig");
pub const where_eval = @import("where_eval.zig");
pub const expr_eval = @import("expr_eval.zig");
pub const group_eval = @import("group_eval.zig");
pub const streaming_reader = @import("streaming_reader.zig");
pub const late_materialization = @import("late_materialization.zig");

// Fused query compilation (optional)
pub const planner = @import("planner/planner.zig");
pub const plan_nodes = @import("planner/plan_nodes.zig");
pub const fused_codegen = @import("codegen/fused_codegen.zig");
const metal0_jit = @import("lanceql.codegen");
const runtime_columns = @import("runtime_columns.zig");
const RuntimeColumns = runtime_columns.RuntimeColumns;
const ColumnDataPtr = runtime_columns.ColumnDataPtr;

const Expr = ast.Expr;
const SelectStmt = ast.SelectStmt;
const Value = ast.Value;
const BinaryOp = ast.BinaryOp;

/// Aggregate types (re-exported from aggregate_functions module)
pub const AggregateType = aggregate_functions.AggregateType;
pub const Accumulator = aggregate_functions.Accumulator;
pub const PercentileAccumulator = aggregate_functions.PercentileAccumulator;

/// Result types (re-exported from result_types module)
pub const Result = result_types.Result;
pub const CachedColumn = result_types.CachedColumn;
pub const JoinedData = result_types.JoinedData;
pub const TableSource = result_types.TableSource;
pub const LanceColumnType = result_types.LanceColumnType;

/// Cached compiled query - holds JIT context and compiled function
const CompiledQuery = struct {
    jit_ctx: metal0_jit.JitContext,
    compiled_ptr: ?*const anyopaque,
    input_columns: []fused_codegen.ColumnInfo,
    output_columns: []fused_codegen.ColumnInfo,
    needs_string_arena: bool,

    fn deinit(self: *CompiledQuery, allocator: std.mem.Allocator) void {
        self.jit_ctx.deinit();
        allocator.free(self.input_columns);
        allocator.free(self.output_columns);
    }
};

/// SQL Query Executor
pub const Executor = struct {
    /// Default table (used when FROM is a simple table name or not specified)
    table: ?*Table,
    /// Parquet table (alternative to Lance table)
    parquet_table: ?*ParquetTable = null,
    /// Delta table (alternative to Lance table)
    delta_table: ?*DeltaTable = null,
    /// Iceberg table (alternative to Lance table)
    iceberg_table: ?*IcebergTable = null,
    /// Arrow IPC table (alternative to Lance table)
    arrow_table: ?*ArrowTable = null,
    /// Avro table (alternative to Lance table)
    avro_table: ?*AvroTable = null,
    /// ORC table (alternative to Lance table)
    orc_table: ?*OrcTable = null,
    /// XLSX table (alternative to Lance table)
    xlsx_table: ?*XlsxTable = null,
    allocator: std.mem.Allocator,
    column_cache: std.StringHashMap(CachedColumn),
    /// Optional dispatcher for @logic_table method calls
    dispatcher: ?*logic_table_dispatch.Dispatcher = null,
    /// Maps table alias to class name for @logic_table instances
    logic_table_aliases: std.StringHashMap([]const u8),
    /// Currently active table source (set during execute)
    active_source: ?TableSource = null,
    /// Registered tables by name (for JOINs and multi-table queries)
    tables: std.StringHashMap(*Table),
    /// Cache for @logic_table method batch results
    /// Key: "ClassName.methodName", Value: results array
    method_results_cache: std.StringHashMap([]const f64),
    /// Cache for compiled queries - key is query hash
    compiled_query_cache: std.AutoHashMap(u64, CompiledQuery),

    /// Enable compiled query execution (fused compilation)
    compiled_execution_enabled: bool = true,

    /// Minimum row count to trigger compilation (below this, interpretation is faster)
    compile_threshold: usize = 10_000,

    /// Vector index metadata storage
    /// Key: "table_name.column_name", Value: VectorIndexInfo
    vector_indexes: std.StringHashMap(VectorIndexInfo),

    /// Dataset path for accessing _versions/ manifests
    /// Set via setDatasetPath() after initialization
    dataset_path: ?[]const u8 = null,

    /// Current SELECT items for projection pushdown in JOINs
    /// Set during execute() before resolving table sources
    current_select_items: ?[]const ast.SelectItem = null,

    const Self = @This();

    /// Vector index metadata
    pub const VectorIndexInfo = struct {
        table_name: []const u8,
        column_name: []const u8,
        model: []const u8,
        shadow_column: []const u8, // e.g., "__vec_text_minilm"
        dimension: u32,
        created_at: i64,
    };

    pub fn init(table: ?*Table, allocator: std.mem.Allocator) Self {
        return .{
            .table = table,
            .parquet_table = null,
            .delta_table = null,
            .iceberg_table = null,
            .arrow_table = null,
            .avro_table = null,
            .orc_table = null,
            .xlsx_table = null,
            .allocator = allocator,
            .column_cache = std.StringHashMap(CachedColumn).init(allocator),
            .dispatcher = null,
            .logic_table_aliases = std.StringHashMap([]const u8).init(allocator),
            .active_source = null,
            .tables = std.StringHashMap(*Table).init(allocator),
            .method_results_cache = std.StringHashMap([]const f64).init(allocator),
            .compiled_query_cache = std.AutoHashMap(u64, CompiledQuery).init(allocator),
            .vector_indexes = std.StringHashMap(VectorIndexInfo).init(allocator),
        };
    }

    /// Initialize executor with a Parquet table
    pub fn initWithParquet(parquet_table: *ParquetTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.parquet_table = parquet_table;
        return self;
    }

    /// Initialize executor with a Delta table
    pub fn initWithDelta(delta_table: *DeltaTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.delta_table = delta_table;
        return self;
    }

    /// Initialize executor with an Iceberg table
    pub fn initWithIceberg(iceberg_table: *IcebergTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.iceberg_table = iceberg_table;
        return self;
    }

    /// Initialize executor with an Arrow IPC table
    pub fn initWithArrow(arrow_table: *ArrowTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.arrow_table = arrow_table;
        return self;
    }

    /// Initialize executor with an Avro table
    pub fn initWithAvro(avro_table: *AvroTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.avro_table = avro_table;
        return self;
    }

    /// Initialize executor with an ORC table
    pub fn initWithOrc(orc_table: *OrcTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.orc_table = orc_table;
        return self;
    }

    /// Initialize executor with an XLSX table
    pub fn initWithXlsx(xlsx_table: *XlsxTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        self.xlsx_table = xlsx_table;
        return self;
    }

    /// Initialize executor with any table type via AnyTable union
    pub fn initWithAnyTable(any_table: *AnyTable, allocator: std.mem.Allocator) Self {
        var self = init(null, allocator);
        switch (any_table.*) {
            .lance => |*t| self.table = t,
            .parquet => |*t| self.parquet_table = t,
            .delta => |*t| self.delta_table = t,
            .iceberg => |*t| self.iceberg_table = t,
            .arrow => |*t| self.arrow_table = t,
            .avro => |*t| self.avro_table = t,
            .orc => |*t| self.orc_table = t,
            .xlsx => |*t| self.xlsx_table = t,
        }
        return self;
    }

    /// Set the dataset path for manifest access (required for DIFF/SHOW VERSIONS)
    /// The path should be the root of the Lance dataset (e.g., "data.lance")
    /// not a specific .lance fragment file.
    pub fn setDatasetPath(self: *Self, path: []const u8) void {
        self.dataset_path = path;
    }

    // ========================================================================
    // Compiled Query Execution (Fused Compilation)
    // ========================================================================

    /// Enable compiled query execution
    /// When enabled, queries that exceed the compile threshold will be JIT compiled
    /// to native code for maximum performance.
    pub fn enableCompiledExecution(self: *Self, enabled: bool) void {
        self.compiled_execution_enabled = enabled;
    }

    /// Set the row count threshold for triggering compilation
    /// Queries with fewer rows than this will use interpretation (faster for small data)
    pub fn setCompileThreshold(self: *Self, threshold: usize) void {
        self.compile_threshold = threshold;
    }

    /// Check if a query should use compiled execution
    /// Returns true if compiled execution is enabled
    /// All queries now use compiled execution - no interpreted fallback
    fn shouldCompile(self: *Self, stmt: *const SelectStmt) bool {
        _ = stmt;
        return self.compiled_execution_enabled;
    }

    /// Check if expression contains @logic_table method call
    fn hasLogicTableMethodCall(self: *Self, expr: *const Expr) bool {
        switch (expr.*) {
            .method_call => |mc| {
                // Check if object is a logic_table alias
                if (self.logic_table_aliases.get(mc.object)) |_| {
                    return true;
                }
                return false;
            },
            .binary => |bin| {
                return self.hasLogicTableMethodCall(bin.left) or
                    self.hasLogicTableMethodCall(bin.right);
            },
            .unary => |un| {
                return self.hasLogicTableMethodCall(un.operand);
            },
            .call => |call| {
                for (call.args) |*arg| {
                    if (self.hasLogicTableMethodCall(arg)) return true;
                }
                return false;
            },
            else => return false,
        }
    }

    /// Execute a query using fused compilation
    /// This generates and JIT-compiles a single function for the entire query
    fn executeCompiled(self: *Self, stmt: *const SelectStmt, params: []const Value) !Result {
        // If we're querying a joined table, use the interpreted path for now
        // (the join has already been materialized in resolveTableSource)
        if (self.active_source) |source| {
            if (source == .joined) {
                // Fall back to executing on the joined data directly
                return self.executeOnJoinedTable(stmt, params, source.joined);
            }
        }

        // 1. Plan the query
        var planner_inst = planner.Planner.init(self.allocator);
        defer planner_inst.deinit();
        const plan = try planner_inst.plan(stmt);

        // Check if plan is compilable (has inlined bodies, no unsupported ops)
        if (!plan.compilable) return error.NotImplemented;

        // 2. Build type and nullable maps from table schema
        var type_map = self.buildSchemaTypeMap() catch return error.NoSchema;
        defer type_map.deinit();
        var nullable_map = self.buildSchemaNullableMap() catch return error.NoSchema;
        defer nullable_map.deinit();

        // 3. Generate fused Zig code with layout metadata and resolved types
        var codegen = fused_codegen.FusedCodeGen.init(self.allocator);
        defer codegen.deinit(); // This handles freeing the generated source code
        codegen.setParams(params);

        // 3a. Check parameter bounds before generating code
        const max_param = findMaxParamIndex(stmt);
        if (max_param) |idx| {
            if (idx >= params.len) return error.ParameterOutOfBounds;
        }

        // 3b. Pre-execute any subqueries and pass results to codegen
        try self.preExecuteSubqueries(stmt, &codegen);

        // 3b. Analyze SELECT and WHERE expressions to extract column references
        // This populates input_columns for code generation
        for (stmt.columns) |col| {
            try codegen.analyzeSelectExpr(&col.expr);
        }
        if (stmt.where) |where| {
            try codegen.analyzeSelectExpr(&where);
        }

        // 3c. For SELECT *, add all table columns if none were found (in schema order)
        // Only do this if the query is actually SELECT * (not aggregate-only queries like COUNT(*))
        const is_select_star = blk: {
            if (stmt.columns.len != 1) break :blk false;
            switch (stmt.columns[0].expr) {
                .column => |col| break :blk std.mem.eql(u8, col.name, "*"),
                else => break :blk false,
            }
        };
        if (!codegen.hasInputColumns() and is_select_star) {
            // For SELECT *, add all columns from the type map (which was built from table schema)
            const col_names_result = self.getColumnNamesWithOwnership() catch return error.NoSchema;
            defer col_names_result.deinit(self.allocator);
            for (col_names_result.names) |name| {
                if (type_map.get(name)) |col_type| {
                    try codegen.addInputColumn(name, col_type);
                }
            }
        }

        // 3d. Register SELECT expressions for inline computation
        // Supports arithmetic expressions and function calls (LENGTH, ABS, UPPER, LOWER, etc.)
        if (!is_select_star and !codegen.hasGroupBy() and !codegen.hasWindowSpecs()) {
            var has_computation = false;
            for (stmt.columns) |col| {
                if (isArithmeticExpr(&col.expr) or hasFunctionCall(&col.expr)) {
                    has_computation = true;
                }
            }
            // Use select_exprs for arithmetic and function calls
            if (has_computation) {
                for (stmt.columns, 0..) |_, idx| {
                    // IMPORTANT: Use indexed access to get stable pointer to expression
                    // Using &col.expr would point to loop variable (same address each iteration)
                    const col_ptr = &stmt.columns[idx];
                    const name = col_ptr.alias orelse blk: {
                        // Generate name for expressions without alias
                        var name_buf: [32]u8 = undefined;
                        break :blk std.fmt.bufPrint(&name_buf, "col_{d}", .{idx}) catch "col";
                    };
                    const col_type = inferExprType(&col_ptr.expr, &type_map);
                    try codegen.addSelectExpr(name, &col_ptr.expr, col_type);
                }
            }
        }

        var gen_result = try codegen.generateWithLayoutTypesAndNullability(&plan, &type_map, &nullable_map);
        const needs_string_arena = codegen.needsStringArena();

        // Hash the generated source to use as cache key
        const query_hash = std.hash.Wyhash.hash(0, gen_result.source);

        // Check compiled query cache
        const cached_query = self.compiled_query_cache.get(query_hash);

        // Get compiled function - either from cache or compile new
        const compiled_ptr: *const anyopaque = if (cached_query) |cached| blk: {
            // Cache hit - free the gen_result layout since we'll use cached layout
            gen_result.layout.deinit();
            break :blk cached.compiled_ptr orelse return error.CompiledFunctionNull;
        } else blk: {
            // Cache miss - compile and store
            var jit_ctx = metal0_jit.JitContext.init(self.allocator);
            errdefer jit_ctx.deinit();

            const compiled = try jit_ctx.compileZigSource(gen_result.source, "fused_query");

            // Copy layout info for cache (gen_result.layout will be freed after this function)
            const input_cols = try self.allocator.dupe(fused_codegen.ColumnInfo, gen_result.layout.input_columns);
            errdefer self.allocator.free(input_cols);
            const output_cols = try self.allocator.dupe(fused_codegen.ColumnInfo, gen_result.layout.output_columns);
            errdefer self.allocator.free(output_cols);

            // Store in cache
            try self.compiled_query_cache.put(query_hash, .{
                .jit_ctx = jit_ctx,
                .compiled_ptr = compiled.ptr,
                .input_columns = input_cols,
                .output_columns = output_cols,
                .needs_string_arena = needs_string_arena,
            });

            // Free gen_result layout since we copied it
            gen_result.layout.deinit();

            break :blk compiled.ptr orelse return error.CompiledFunctionNull;
        };

        // Get layout from cache (guaranteed to exist now)
        const cached = self.compiled_query_cache.get(query_hash).?;
        const input_columns = cached.input_columns;
        const output_columns = cached.output_columns;

        // 4. Load column data based on layout
        const row_count = try self.getRowCount();
        const column_data = try self.loadColumnDataForLayout(input_columns);
        defer self.freeCompiledColumnData(column_data);

        // 4b. Load validity bitmaps for nullable columns
        const validity_bitmaps = try self.loadValidityBitmapsForLayout(input_columns, row_count);
        defer self.freeValidityBitmaps(validity_bitmaps);

        // 5. Build runtime columns struct (input) with validity bitmaps
        var columns = try RuntimeColumns.buildInputWithValidity(
            self.allocator,
            input_columns,
            column_data,
            validity_bitmaps,
            row_count,
        );
        defer columns.deinit();

        // 6. Allocate output buffers
        var output = try RuntimeColumns.buildOutput(
            self.allocator,
            output_columns,
            row_count,
        );
        defer output.deinit();

        // 7. Allocate string arena if needed (must outlive output extraction)
        const string_arena: ?[]u8 = if (cached.needs_string_arena)
            try self.allocator.alloc(u8, row_count * 1024) // 1KB per row
        else
            null;
        defer if (string_arena) |arena| self.allocator.free(arena);

        // 8. Call compiled function
        const result_count = if (string_arena) |arena| blk: {
            const FusedQueryFnWithArena = *const fn (*anyopaque, *anyopaque, [*]u8) callconv(std.builtin.CallingConvention.c) usize;
            const func: FusedQueryFnWithArena = @ptrCast(@alignCast(compiled_ptr));
            break :blk func(columns.asPtr(), output.asPtr(), arena.ptr);
        } else blk: {
            const FusedQueryFn = *const fn (*anyopaque, *anyopaque) callconv(std.builtin.CallingConvention.c) usize;
            const func: FusedQueryFn = @ptrCast(@alignCast(compiled_ptr));
            break :blk func(columns.asPtr(), output.asPtr());
        };

        // 9. Convert output to Result
        return self.outputToResult(&output, result_count, stmt.columns);
    }

    /// Load column data based on layout for compiled execution
    fn loadColumnDataForLayout(
        self: *Self,
        layout: []const fused_codegen.ColumnInfo,
    ) ![]ColumnDataPtr {
        var data = try self.allocator.alloc(ColumnDataPtr, layout.len);
        errdefer self.allocator.free(data);

        for (layout, 0..) |col, i| {
            const col_idx = self.getPhysicalColumnIndex(col.name) orelse return error.ColumnNotFound;
            // Types should already be resolved by generateWithLayoutAndTypes
            data[i] = switch (col.col_type) {
                .i64 => .{ .i64 = try self.readInt64ColumnTyped(col_idx) },
                .f64 => .{ .f64 = try self.readFloat64ColumnTyped(col_idx) },
                .i32 => .{ .i32 = try self.readInt32ColumnTyped(col_idx) },
                .f32 => .{ .f32 = try self.readFloat32ColumnTyped(col_idx) },
                .bool => .{ .bool_ = try self.readBoolColumnTyped(col_idx) },
                .string => .{ .string = try self.readStringColumnTyped(col_idx) },
                .timestamp_ns, .timestamp_us, .timestamp_ms, .timestamp_s => .{ .timestamp_ns = try self.readInt64ColumnTyped(col_idx) },
                .date32 => .{ .date32 = try self.readInt32ColumnTyped(col_idx) },
                .date64 => .{ .date64 = try self.readInt64ColumnTyped(col_idx) },
                .vec_f32 => .{ .vec_f32 = try self.readFloat32ColumnTyped(col_idx) },
                .vec_f64 => .{ .vec_f64 = try self.readFloat64ColumnTyped(col_idx) },
                else => .{ .f64 = try self.readFloat64ColumnTyped(col_idx) }, // Default fallback
            };
        }
        return data;
    }

    /// Load validity bitmaps for columns based on layout
    /// Returns array of optional validity bitmaps (null means column is non-nullable)
    /// Reads actual validity bitmaps from Lance file if column is nullable
    fn loadValidityBitmapsForLayout(
        self: *Self,
        layout: []const fused_codegen.ColumnInfo,
        row_count: usize,
    ) ![]?[]const u8 {
        var validity = try self.allocator.alloc(?[]const u8, layout.len);
        errdefer self.allocator.free(validity);

        const bytes_needed = (row_count + 7) / 8; // Round up to nearest byte

        for (layout, 0..) |col, i| {
            if (col.nullable) {
                // Try to read actual validity bitmap from Lance file
                const col_idx = self.getPhysicalColumnIndex(col.name) orelse {
                    // Column not found - create all-valid bitmap
                    const bitmap = try self.allocator.alloc(u8, bytes_needed);
                    @memset(bitmap, 0xFF);
                    validity[i] = bitmap;
                    continue;
                };

                // Read validity bitmap from Lance table
                const maybe_bitmap = self.tbl().readValidityBitmap(col_idx) catch {
                    // Error reading - create all-valid bitmap as fallback
                    const bitmap = try self.allocator.alloc(u8, bytes_needed);
                    @memset(bitmap, 0xFF);
                    validity[i] = bitmap;
                    continue;
                };

                if (maybe_bitmap) |bitmap| {
                    // Got actual validity bitmap from file
                    validity[i] = bitmap;
                } else {
                    // Column has no validity bitmap (all values valid)
                    const bitmap = try self.allocator.alloc(u8, bytes_needed);
                    @memset(bitmap, 0xFF);
                    validity[i] = bitmap;
                }
            } else {
                validity[i] = null;
            }
        }
        return validity;
    }

    /// Free validity bitmaps loaded by loadValidityBitmapsForLayout
    fn freeValidityBitmaps(self: *Self, validity: []?[]const u8) void {
        for (validity) |maybe_bitmap| {
            if (maybe_bitmap) |bitmap| {
                self.allocator.free(bitmap);
            }
        }
        self.allocator.free(validity);
    }

    /// Read int64 column - handles typed tables (Parquet, Arrow, etc.) and Lance
    fn readInt64ColumnTyped(self: *Self, col_idx: u32) ![]i64 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return t.readInt64Column(col_idx);
            }
        }
        return self.tbl().readInt64Column(col_idx);
    }

    /// Read float64 column - handles typed tables and Lance
    fn readFloat64ColumnTyped(self: *Self, col_idx: u32) ![]f64 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return t.readFloat64Column(col_idx);
            }
        }
        return self.tbl().readFloat64Column(col_idx);
    }

    /// Read int32 column - handles typed tables and Lance
    fn readInt32ColumnTyped(self: *Self, col_idx: u32) ![]i32 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return t.readInt32Column(col_idx);
            }
        }
        return self.tbl().readInt32Column(col_idx);
    }

    /// Read float32 column - handles typed tables and Lance
    fn readFloat32ColumnTyped(self: *Self, col_idx: u32) ![]f32 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return t.readFloat32Column(col_idx);
            }
        }
        return self.tbl().readFloat32Column(col_idx);
    }

    /// Read bool column - handles typed tables and Lance
    fn readBoolColumnTyped(self: *Self, col_idx: u32) ![]bool {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return t.readBoolColumn(col_idx);
            }
        }
        return self.tbl().readBoolColumn(col_idx);
    }

    /// Read string column - handles typed tables and Lance
    fn readStringColumnTyped(self: *Self, col_idx: u32) ![][]const u8 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return t.readStringColumn(col_idx);
            }
        }
        return self.tbl().readStringColumn(col_idx);
    }

    /// Free column data loaded for compiled execution
    fn freeCompiledColumnData(self: *Self, data: []ColumnDataPtr) void {
        for (data) |col| {
            switch (col) {
                .i64 => |s| self.allocator.free(s),
                .i32 => |s| self.allocator.free(s),
                .i16 => |s| self.allocator.free(s),
                .i8 => |s| self.allocator.free(s),
                .u64 => |s| self.allocator.free(s),
                .u32 => |s| self.allocator.free(s),
                .u16 => |s| self.allocator.free(s),
                .u8 => |s| self.allocator.free(s),
                .f64 => |s| self.allocator.free(s),
                .f32 => |s| self.allocator.free(s),
                .bool_ => |s| self.allocator.free(s),
                .string => |s| {
                    // Free each individual string first
                    for (s) |str| self.allocator.free(str);
                    // Then free the outer slice
                    self.allocator.free(s);
                },
                .timestamp_ns => |s| self.allocator.free(s),
                .timestamp_us => |s| self.allocator.free(s),
                .timestamp_ms => |s| self.allocator.free(s),
                .date32 => |s| self.allocator.free(s),
                .date64 => |s| self.allocator.free(s),
                .vec_f32 => |s| self.allocator.free(s),
                .vec_f64 => |s| self.allocator.free(s),
                .empty => {},
            }
        }
        self.allocator.free(data);
    }

    /// Pre-execute subqueries and pass results to codegen
    fn preExecuteSubqueries(self: *Self, stmt: *const SelectStmt, codegen: *fused_codegen.FusedCodeGen) !void {
        // Collect subqueries from WHERE clause
        if (stmt.where) |where| {
            try self.collectAndExecuteSubqueries(&where, codegen);
        }

        // Collect subqueries from SELECT expressions
        for (stmt.columns) |col| {
            try self.collectAndExecuteSubqueries(&col.expr, codegen);
        }
    }

    /// Recursively find and execute subqueries in an expression
    fn collectAndExecuteSubqueries(self: *Self, expr: *const Expr, codegen: *fused_codegen.FusedCodeGen) anyerror!void {
        switch (expr.*) {
            .in_subquery => |in| {
                // Execute the subquery
                var result = try self.execute(in.subquery, &[_]Value{});
                defer result.deinit();

                // Extract integer values from single-column result
                if (result.columns.len == 1) {
                    const col = result.columns[0];
                    var values = try self.allocator.alloc(i64, result.row_count);
                    for (0..result.row_count) |i| {
                        values[i] = switch (col.data) {
                            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| data[i],
                            .int32, .date32 => |data| data[i],
                            .float64 => |data| @as(i64, @intFromFloat(data[i])),
                            .float32 => |data| @as(i64, @intFromFloat(data[i])),
                            .bool_ => |data| if (data[i]) @as(i64, 1) else 0,
                            .string => 0, // String subqueries not supported for IN
                        };
                    }
                    try codegen.setSubqueryIntResults(@intFromPtr(in.subquery), values);
                }
            },
            .exists => |ex| {
                // Execute the subquery and check if rows exist
                var result = try self.execute(ex.subquery, &[_]Value{});
                defer result.deinit();
                try codegen.setExistsResult(@intFromPtr(ex.subquery), result.row_count > 0);
            },
            .binary => |bin| {
                try self.collectAndExecuteSubqueries(bin.left, codegen);
                try self.collectAndExecuteSubqueries(bin.right, codegen);
            },
            .unary => |un| {
                try self.collectAndExecuteSubqueries(un.operand, codegen);
            },
            .call => |call| {
                for (call.args) |*arg| {
                    try self.collectAndExecuteSubqueries(arg, codegen);
                }
            },
            .in_list => |in| {
                try self.collectAndExecuteSubqueries(in.expr, codegen);
            },
            .between => |bet| {
                try self.collectAndExecuteSubqueries(bet.expr, codegen);
                try self.collectAndExecuteSubqueries(bet.low, codegen);
                try self.collectAndExecuteSubqueries(bet.high, codegen);
            },
            .case_expr => |case| {
                if (case.operand) |op| try self.collectAndExecuteSubqueries(op, codegen);
                for (case.when_clauses) |clause| {
                    try self.collectAndExecuteSubqueries(&clause.condition, codegen);
                    try self.collectAndExecuteSubqueries(&clause.result, codegen);
                }
                if (case.else_result) |else_expr| try self.collectAndExecuteSubqueries(else_expr, codegen);
            },
            else => {},
        }
    }

    /// Get physical column index by name
    fn getPhysicalColumnIndex(self: *Self, name: []const u8) ?u32 {
        // Handle typed tables (Parquet, Arrow, etc.)
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return if (t.columnIndex(name)) |idx| @intCast(idx) else null;
            }
        }
        // Handle Lance Table
        const schema = self.tbl().getSchema() orelse return null;
        for (schema.fields, 0..) |field_info, i| {
            if (std.mem.eql(u8, field_info.name, name)) {
                return @intCast(i);
            }
        }
        return null;
    }

    /// Build a type map from the table schema
    fn buildSchemaTypeMap(self: *Self) !std.StringHashMap(plan_nodes.ColumnType) {
        var type_map = std.StringHashMap(plan_nodes.ColumnType).init(self.allocator);
        errdefer type_map.deinit();

        // Handle typed tables (Parquet, Arrow, etc.)
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                const col_names = t.getColumnNames();
                for (col_names, 0..) |name, idx| {
                    const pq_type = t.getColumnType(idx);
                    const col_type: plan_nodes.ColumnType = if (pq_type) |pt| mapParquetType(pt) else .unknown;
                    try type_map.put(name, col_type);
                }
                return type_map;
            }
        }

        // Handle Lance Table
        const schema = self.tbl().getSchema() orelse return error.NoSchema;
        for (schema.fields) |field| {
            const lance_type = LanceColumnType.fromLogicalType(field.logical_type);
            // Map LanceColumnType to plan_nodes.ColumnType
            const col_type: plan_nodes.ColumnType = switch (lance_type) {
                .int64 => .i64,
                .int32 => .i32,
                .float64 => .f64,
                .float32 => .f32,
                .string => .string,
                .bool_ => .bool,
                .timestamp_ns => .timestamp_ns,
                .timestamp_us => .timestamp_us,
                .timestamp_ms => .timestamp_ms,
                .timestamp_s => .timestamp_s,
                .date32 => .date32,
                .date64 => .date64,
                .unsupported => .unknown,
            };
            try type_map.put(field.name, col_type);
        }
        return type_map;
    }

    /// Build a map of column names to their nullable status from table schema
    fn buildSchemaNullableMap(self: *Self) !std.StringHashMap(bool) {
        var nullable_map = std.StringHashMap(bool).init(self.allocator);
        errdefer nullable_map.deinit();

        // Handle typed tables (Parquet, Arrow, etc.)
        // Note: For typed tables, default to non-nullable (conservative)
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                const col_names = t.getColumnNames();
                for (col_names) |name| {
                    try nullable_map.put(name, false);
                }
                return nullable_map;
            }
        }

        // Handle Lance Table - uses schema nullable field
        const schema = self.tbl().getSchema() orelse return error.NoSchema;
        for (schema.fields) |field| {
            try nullable_map.put(field.name, field.nullable);
        }
        return nullable_map;
    }

    /// Map Parquet Type to plan_nodes.ColumnType
    fn mapParquetType(pt: meta.Type) plan_nodes.ColumnType {
        return switch (pt) {
            .int64 => .i64,
            .int32 => .i32,
            .double => .f64,
            .float => .f32,
            .byte_array, .fixed_len_byte_array => .string,
            .boolean => .bool,
            .int96 => .timestamp_ns, // INT96 is typically used for timestamps
            _ => .unknown,
        };
    }

    /// Convert output buffers to Result
    fn outputToResult(
        self: *Self,
        output: *RuntimeColumns,
        result_count: usize,
        select_list: []const ast.SelectItem,
    ) !Result {
        // Handle SELECT * - check if first column is wildcard
        const is_select_star = blk: {
            if (select_list.len != 1) break :blk false;
            switch (select_list[0].expr) {
                .column => |col| break :blk std.mem.eql(u8, col.name, "*"),
                else => break :blk false,
            }
        };

        if (is_select_star) {
            // Use output column names from layout
            var columns = try self.allocator.alloc(Result.Column, output.column_data.len);
            errdefer self.allocator.free(columns);

            for (output.column_names, 0..) |name, i| {
                columns[i] = .{
                    .name = name,
                    .data = try self.extractResultData(output.column_data[i], result_count),
                };
            }

            return Result{
                .columns = columns,
                .row_count = result_count,
                .allocator = self.allocator,
            };
        }

        var columns = try self.allocator.alloc(Result.Column, select_list.len);
        errdefer self.allocator.free(columns);

        for (select_list, 0..) |item, i| {
            const name = item.alias orelse blk: {
                // Extract column name from expression
                if (item.expr == .column) {
                    break :blk item.expr.column.name;
                } else if (item.expr == .method_call) {
                    break :blk item.expr.method_call.method;
                } else {
                    break :blk "?column?";
                }
            };

            // Copy output data up to result_count
            if (i < output.column_data.len) {
                columns[i] = .{
                    .name = name, // Name comes from AST, no need to dupe
                    .data = try self.extractResultData(output.column_data[i], result_count),
                };
            } else {
                // Column not in output - should not happen
                columns[i] = .{
                    .name = name, // Name comes from AST, no need to dupe
                    .data = .{ .float64 = &[_]f64{} },
                };
            }
        }

        return Result{
            .columns = columns,
            .row_count = result_count,
            .allocator = self.allocator,
        };
    }

    /// Extract result data from column data pointer
    fn extractResultData(self: *Self, col: ColumnDataPtr, count: usize) !Result.ColumnData {
        return switch (col) {
            .i64 => |s| .{ .int64 = try self.allocator.dupe(i64, s[0..count]) },
            .i32 => |s| .{ .int32 = try self.allocator.dupe(i32, s[0..count]) },
            .f64 => |s| .{ .float64 = try self.allocator.dupe(f64, s[0..count]) },
            .f32 => |s| .{ .float32 = try self.allocator.dupe(f32, s[0..count]) },
            .bool_ => |s| .{ .bool_ = try self.allocator.dupe(bool, s[0..count]) },
            .string => |s| blk: {
                var result = try self.allocator.alloc([]const u8, count);
                errdefer self.allocator.free(result);
                var allocated: usize = 0;
                errdefer for (result[0..allocated]) |str| self.allocator.free(str);
                for (s[0..count], 0..) |str, i| {
                    result[i] = try self.allocator.dupe(u8, str);
                    allocated = i + 1;
                }
                break :blk .{ .string = result };
            },
            .timestamp_ns, .timestamp_us, .timestamp_ms => |s| .{ .timestamp_ns = try self.allocator.dupe(i64, s[0..count]) },
            .date32 => |s| .{ .date32 = try self.allocator.dupe(i32, s[0..count]) },
            .date64 => |s| .{ .date64 = try self.allocator.dupe(i64, s[0..count]) },
            .vec_f32 => |s| .{ .float32 = try self.allocator.dupe(f32, s[0..count]) },
            .vec_f64 => |s| .{ .float64 = try self.allocator.dupe(f64, s[0..count]) },
            else => .{ .float64 = &[_]f64{} },
        };
    }

    /// Register a table by name for use in JOINs and multi-table queries
    pub fn registerTable(self: *Self, name: []const u8, table: *Table) !void {
        try self.tables.put(name, table);
    }

    /// Get a registered table by name
    pub fn getRegisteredTable(self: *Self, name: []const u8) ?*Table {
        return self.tables.get(name);
    }

    /// Initialize with a table (convenience for existing code)
    pub fn initWithTable(table: *Table, allocator: std.mem.Allocator) Self {
        return init(table, allocator);
    }

    /// Get the table (must be set before calling this)
    /// This is used by internal methods that expect a table to be configured.
    inline fn tbl(self: *Self) *Table {
        return self.table orelse unreachable;
    }

    /// Check if operating in Parquet mode
    inline fn isParquetMode(self: *Self) bool {
        return self.parquet_table != null;
    }

    /// Check if operating in Delta mode
    inline fn isDeltaMode(self: *Self) bool {
        return self.delta_table != null;
    }

    /// Check if operating in Iceberg mode
    inline fn isIcebergMode(self: *Self) bool {
        return self.iceberg_table != null;
    }

    /// Check if operating in Arrow mode
    inline fn isArrowMode(self: *Self) bool {
        return self.arrow_table != null;
    }

    /// Check if operating in Avro mode
    inline fn isAvroMode(self: *Self) bool {
        return self.avro_table != null;
    }

    /// Check if operating in ORC mode
    inline fn isOrcMode(self: *Self) bool {
        return self.orc_table != null;
    }

    /// Check if operating in XLSX mode
    inline fn isXlsxMode(self: *Self) bool {
        return self.xlsx_table != null;
    }

    /// Field names of typed table pointers for comptime iteration
    const typed_table_fields = .{ "parquet_table", "delta_table", "iceberg_table", "arrow_table", "avro_table", "orc_table", "xlsx_table" };

    /// Check if any typed table is set (non-Lance mode)
    inline fn hasTypedTable(self: *Self) bool {
        inline for (typed_table_fields) |field| {
            if (@field(self, field) != null) return true;
        }
        return false;
    }

    /// Get row count (works with Lance, Parquet, Delta, Iceberg, Arrow, Avro, ORC, or XLSX)
    fn getRowCount(self: *Self) !usize {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return t.numRows();
        }
        return try self.tbl().rowCount(0);
    }

    /// Column names result with ownership info
    const ColumnNamesResult = struct {
        names: [][]const u8,
        owned: bool, // If true, caller must free with freeColumnNames

        fn deinit(self: @This(), allocator: std.mem.Allocator) void {
            if (self.owned) {
                allocator.free(self.names);
            }
        }
    };

    /// Get column names (works with Lance, Parquet, Delta, Iceberg, Arrow, Avro, ORC, or XLSX)
    /// Returns ownership info - caller must call deinit() on result
    fn getColumnNamesWithOwnership(self: *Self) !ColumnNamesResult {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return .{ .names = t.getColumnNames(), .owned = false };
        }
        return .{ .names = try self.tbl().columnNames(), .owned = true };
    }


    /// Get physical column ID by name (works with Lance, Parquet, Delta, Iceberg, Arrow, Avro, ORC, or XLSX)
    fn getPhysicalColumnId(self: *Self, name: []const u8) ?u32 {
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| {
                return if (t.columnIndex(name)) |idx| @intCast(idx) else null;
            }
        }
        return self.tbl().physicalColumnId(name);
    }

    /// Filter array by indices - comptime generic for DRY
    fn filterByIndices(self: *Self, comptime T: type, all_data: []const T, indices: []const u32) ![]T {
        const filtered = try self.allocator.alloc(T, indices.len);
        for (indices, 0..) |idx, i| filtered[i] = all_data[idx];
        return filtered;
    }

    /// Method call wrapper conforming to MethodCallFn signature
    /// Bridges expr_eval to executor's evaluateMethodCall
    fn methodCallWrapper(ctx: *anyopaque, expr: *const Expr, row_idx: u32) anyerror!Value {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return switch (expr.*) {
            .method_call => |mc| self.evaluateMethodCall(mc, row_idx),
            else => error.NotAMethodCall,
        };
    }

    /// Build ExprContext for expr_eval module
    fn buildExprContext(self: *Self) expr_eval.ExprContext {
        return .{
            .allocator = self.allocator,
            .column_cache = &self.column_cache,
            .method_ctx = self,
            .method_call_fn = methodCallWrapper,
        };
    }

    /// Build GroupContext for group_eval module
    fn buildGroupContext(self: *Self) group_eval.GroupContext {
        return .{
            .allocator = self.allocator,
            .column_cache = &self.column_cache,
        };
    }

    /// Set the dispatcher for @logic_table method calls
    pub fn setDispatcher(self: *Self, dispatcher: *logic_table_dispatch.Dispatcher) void {
        self.dispatcher = dispatcher;
    }

    /// Register a @logic_table alias with its class name
    /// Returns error.DuplicateAlias if alias already registered
    pub fn registerLogicTableAlias(self: *Self, alias: []const u8, class_name: []const u8) !void {
        // Check for existing alias first - we don't support overwriting
        if (self.logic_table_aliases.contains(alias)) {
            return error.DuplicateAlias;
        }

        const alias_copy = try self.allocator.dupe(u8, alias);
        errdefer self.allocator.free(alias_copy);
        const class_copy = try self.allocator.dupe(u8, class_name);
        errdefer self.allocator.free(class_copy);
        try self.logic_table_aliases.put(alias_copy, class_copy);
    }

    pub fn deinit(self: *Self) void {
        // Free cached columns
        var iter = self.column_cache.valueIterator();
        while (iter.next()) |col| {
            col.free(self.allocator);
        }
        self.column_cache.deinit();

        // Free logic_table alias keys and values
        var alias_iter = self.logic_table_aliases.iterator();
        while (alias_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.logic_table_aliases.deinit();

        // Free method results cache
        var results_iter = self.method_results_cache.iterator();
        while (results_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.method_results_cache.deinit();

        // Free compiled query cache
        var cache_iter = self.compiled_query_cache.valueIterator();
        while (cache_iter.next()) |cached| {
            var query = cached;
            query.deinit(self.allocator);
        }
        self.compiled_query_cache.deinit();

        // Clean up registered tables map (tables are owned by caller, just deinit the map)
        self.tables.deinit();
    }

    // ========================================================================
    // Column Preloading (for WHERE clause optimization)
    // ========================================================================

    /// Extract all column names referenced in an expression
    fn extractColumnNames(self: *Self, expr: *const Expr, list: *std.ArrayList([]const u8)) anyerror!void {
        switch (expr.*) {
            .column => |col| {
                try list.append(self.allocator, col.name);
            },
            .binary => |bin| {
                try self.extractColumnNames(bin.left, list);
                try self.extractColumnNames(bin.right, list);
            },
            .unary => |un| {
                try self.extractColumnNames(un.operand, list);
            },
            .in_list => |in| {
                try self.extractColumnNames(in.expr, list);
                for (in.values) |*val| {
                    try self.extractColumnNames(val, list);
                }
            },
            .in_subquery => |in| {
                try self.extractColumnNames(in.expr, list);
                // Don't extract from subquery - it has its own scope
            },
            .call => |call| {
                for (call.args) |*arg| {
                    try self.extractColumnNames(arg, list);
                }
            },
            else => {},
        }
    }

    /// Preload columns into cache
    fn preloadColumns(self: *Self, col_names: []const []const u8) !void {
        // Use generic preloader for typed tables
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return self.preloadColumnsFromTable(t, col_names);
        }

        for (col_names) |name| {
            if (self.column_cache.contains(name)) continue;

            const physical_col_id = self.tbl().physicalColumnId(name) orelse return error.ColumnNotFound;
            const fld = self.tbl().getFieldById(physical_col_id) orelse return error.InvalidColumn;
            const col_type = LanceColumnType.fromLogicalType(fld.logical_type);

            const cached = switch (col_type) {
                .timestamp_ns => CachedColumn{ .timestamp_ns = try self.tbl().readInt64Column(physical_col_id) },
                .timestamp_us => CachedColumn{ .timestamp_us = try self.tbl().readInt64Column(physical_col_id) },
                .timestamp_ms => CachedColumn{ .timestamp_ms = try self.tbl().readInt64Column(physical_col_id) },
                .timestamp_s => CachedColumn{ .timestamp_s = try self.tbl().readInt64Column(physical_col_id) },
                .date32 => CachedColumn{ .date32 = try self.tbl().readInt32Column(physical_col_id) },
                .date64 => CachedColumn{ .date64 = try self.tbl().readInt64Column(physical_col_id) },
                .int32 => CachedColumn{ .int32 = try self.tbl().readInt32Column(physical_col_id) },
                .float32 => CachedColumn{ .float32 = try self.tbl().readFloat32Column(physical_col_id) },
                .bool_ => CachedColumn{ .bool_ = try self.tbl().readBoolColumn(physical_col_id) },
                .int64 => CachedColumn{ .int64 = try self.tbl().readInt64Column(physical_col_id) },
                .float64 => CachedColumn{ .float64 = try self.tbl().readFloat64Column(physical_col_id) },
                .string => CachedColumn{ .string = try self.tbl().readStringColumn(physical_col_id) },
                .unsupported => return error.UnsupportedColumnType,
            };
            try self.column_cache.put(name, cached);
        }
    }

    /// Generic column preloader for any table type with standard interface
    fn preloadColumnsFromTable(self: *Self, table: anytype, col_names: []const []const u8) !void {
        const T = @TypeOf(table.*);
        const is_xlsx = T == XlsxTable;

        for (col_names) |name| {
            if (self.column_cache.contains(name)) continue;

            const col_idx = table.columnIndex(name) orelse return error.ColumnNotFound;
            const col_type = table.getColumnType(col_idx) orelse return error.InvalidColumn;

            switch (col_type) {
                .int64 => {
                    const data = table.readInt64Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .int64 = data });
                },
                .int32 => {
                    const data = table.readInt32Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .int32 = data });
                },
                .double => {
                    const data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .float64 = data });
                },
                .float => {
                    const data = table.readFloat32Column(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .float32 = data });
                },
                .boolean => {
                    const data = table.readBoolColumn(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .bool_ = data });
                },
                .byte_array, .fixed_len_byte_array => {
                    const data = table.readStringColumn(col_idx) catch return error.ColumnReadError;
                    try self.column_cache.put(name, CachedColumn{ .string = data });
                },
                else => {
                    // XLSX defaults to float64, others error
                    if (is_xlsx) {
                        const data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                        try self.column_cache.put(name, CachedColumn{ .float64 = data });
                    } else {
                        return error.UnsupportedColumnType;
                    }
                },
            }
        }
    }

    /// Get the currently active table for query execution
    fn getActiveTable(self: *Self) !*Table {
        if (self.active_source) |source| {
            return source.getTable();
        }
        return self.table orelse error.NoTableConfigured;
    }

    /// Resolve FROM clause to get the table source
    fn resolveTableSource(self: *Self, from: *const ast.TableRef) anyerror!TableSource {
        switch (from.*) {
            .simple => |simple| {
                // First check if table is registered by name
                if (self.tables.get(simple.name)) |registered_table| {
                    return .{ .direct = registered_table };
                }
                // Otherwise use the default table
                const direct_table = self.table orelse return error.NoTableConfigured;
                return .{ .direct = direct_table };
            },
            .function => |func| {
                // Table-valued function (e.g., logic_table('fraud.py'))
                if (std.mem.eql(u8, func.func.name, "logic_table")) {
                    // Extract file path from first argument
                    if (func.func.args.len == 0) {
                        return error.LogicTableRequiresPath;
                    }

                    const path_arg = func.func.args[0];
                    const path = switch (path_arg) {
                        .value => |val| switch (val) {
                            .string => |s| s,
                            else => return error.LogicTablePathMustBeString,
                        },
                        else => return error.LogicTablePathMustBeString,
                    };

                    // Create LogicTableExecutor from file path (heap allocated)
                    const executor = try self.allocator.create(logic_table_dispatch.LogicTableExecutor);
                    errdefer self.allocator.destroy(executor);
                    executor.* = try logic_table_dispatch.LogicTableExecutor.init(self.allocator, path);
                    errdefer executor.deinit();

                    // Load tables referenced in the Python file
                    try executor.loadTables();

                    // Get primary table (first loaded table)
                    const primary_table = executor.getPrimaryTable() orelse {
                        executor.deinit();
                        self.allocator.destroy(executor);
                        return error.NoTablesInLogicTable;
                    };

                    // Register alias for method dispatch
                    if (func.alias) |alias| {
                        try self.registerLogicTableAlias(alias, executor.class_name);
                    }

                    return .{ .logic_table = .{
                        .executor = executor,
                        .primary_table = primary_table,
                        .alias = func.alias,
                    } };
                }
                return error.UnsupportedTableFunction;
            },
            .join => |join| {
                // Execute JOIN by resolving both sides and performing hash join
                return try self.executeJoin(join.left, &join.join_clause);
            },
        }
    }

    /// Execute a JOIN operation using hash join algorithm
    fn executeJoin(self: *Self, left_ref: *const ast.TableRef, join_clause: *const ast.JoinClause) !TableSource {
        var left_source = try self.resolveTableSource(left_ref);
        errdefer self.releaseTableSource(&left_source);
        const left_table = left_source.getTable();

        var right_source = try self.resolveTableSource(join_clause.table);
        defer self.releaseTableSource(&right_source);
        const right_table = right_source.getTable();

        // Get table names for vector index resolution
        const left_table_name = getTableRefName(left_ref);
        const right_table_name = getTableRefName(join_clause.table);

        const join_keys = try self.extractJoinKeys(
            join_clause.on_condition orelse return error.JoinRequiresOnCondition,
            left_table_name,
            right_table_name,
        );

        const left_key_col_idx = left_table.physicalColumnId(join_keys.left_col) orelse return error.JoinColumnNotFound;
        const right_key_col_idx = right_table.physicalColumnId(join_keys.right_col) orelse return error.JoinColumnNotFound;

        const left_key_data = try self.readJoinKeyColumn(left_table, left_key_col_idx);
        defer self.freeJoinKeyData(left_key_data);

        const right_key_data = try self.readJoinKeyColumn(right_table, right_key_col_idx);
        defer self.freeJoinKeyData(right_key_data);

        // Pre-allocate output lists (estimate: 1:1 match ratio)
        var left_indices = std.ArrayListUnmanaged(usize){};
        defer left_indices.deinit(self.allocator);
        var right_indices = std.ArrayListUnmanaged(usize){};
        defer right_indices.deinit(self.allocator);

        // Use LinearHashTable for int64 keys (fast path with SIMD hashing)
        // Fall back to StringHashMap for other types
        if (left_key_data == .int64 and right_key_data == .int64) {
            // Fast path: DuckDB-style linear probing hash table with SIMD
            const left_keys = left_key_data.int64;
            const right_keys = right_key_data.int64;

            var ht = try vector_engine.LinearHashTable.init(self.allocator, right_keys.len);
            defer ht.deinit(self.allocator);

            // Build phase: hash right table keys using SIMD
            ht.buildFromColumn(right_keys);

            const is_inner_join = join_clause.join_type == .inner;

            if (is_inner_join) {
                // Fast path: Use JoinHashTable for large tables (better parallel scaling)
                // or LinearHashTable for smaller tables (better cache locality)
                if (left_keys.len >= gpu_hash_join.JOIN_THRESHOLD) {
                    // Large table: use JoinHashTable with parallel probe
                    const join_result = try gpu_hash_join.hashJoinI64(self.allocator, right_keys, left_keys);
                    defer self.allocator.free(join_result.build_indices);
                    defer self.allocator.free(join_result.probe_indices);

                    // Use appendSlice for safe bulk copy
                    try left_indices.appendSlice(self.allocator, join_result.probe_indices[0..join_result.count]);
                    try right_indices.appendSlice(self.allocator, join_result.build_indices[0..join_result.count]);
                } else {
                    // Small table: use LinearHashTable with parallel probe
                    const result = try ht.probeAllParallel(self.allocator, left_keys);
                    defer self.allocator.free(result.left_indices);
                    defer self.allocator.free(result.right_indices);

                    // Use appendSlice for safe bulk copy
                    try left_indices.appendSlice(self.allocator, result.left_indices);
                    try right_indices.appendSlice(self.allocator, result.right_indices);
                }
            } else {
                // Outer join: single-threaded with bitmap tracking
                try left_indices.ensureTotalCapacity(self.allocator, left_keys.len);
                try right_indices.ensureTotalCapacity(self.allocator, left_keys.len);

                var matched_left_bitmap: ?[]bool = null;
                var matched_right_bitmap: ?[]bool = null;
                defer if (matched_left_bitmap) |b| self.allocator.free(b);
                defer if (matched_right_bitmap) |b| self.allocator.free(b);

                if (join_clause.join_type == .left or join_clause.join_type == .full) {
                    matched_left_bitmap = try self.allocator.alloc(bool, left_keys.len);
                    @memset(matched_left_bitmap.?, false);
                }
                if (join_clause.join_type == .right or join_clause.join_type == .full) {
                    matched_right_bitmap = try self.allocator.alloc(bool, right_keys.len);
                    @memset(matched_right_bitmap.?, false);
                }

                const est_batch_matches = vector_engine.VECTOR_SIZE * 4;
                const left_batch_out = try self.allocator.alloc(usize, est_batch_matches);
                defer self.allocator.free(left_batch_out);
                const right_batch_out = try self.allocator.alloc(usize, est_batch_matches);
                defer self.allocator.free(right_batch_out);

                var offset: usize = 0;
                const batch_size = vector_engine.VECTOR_SIZE;

                while (offset < left_keys.len) {
                    const batch_end = @min(offset + batch_size, left_keys.len);
                    const batch_keys = left_keys[offset..batch_end];

                    const matches = ht.probeBatch(batch_keys, right_keys, left_batch_out, right_batch_out, offset);

                    try left_indices.ensureUnusedCapacity(self.allocator, matches);
                    try right_indices.ensureUnusedCapacity(self.allocator, matches);

                    for (0..matches) |m| {
                        left_indices.appendAssumeCapacity(left_batch_out[m]);
                        right_indices.appendAssumeCapacity(right_batch_out[m]);

                        if (matched_left_bitmap) |b| b[left_batch_out[m]] = true;
                        if (matched_right_bitmap) |b| b[right_batch_out[m]] = true;
                    }

                    offset = batch_end;
                }

                // LEFT/FULL JOIN: add unmatched left rows
                if (matched_left_bitmap) |bitmap| {
                    for (bitmap, 0..) |matched, left_idx| {
                        if (!matched) {
                            try left_indices.append(self.allocator, left_idx);
                            try right_indices.append(self.allocator, std.math.maxInt(usize));
                        }
                    }
                }

                // RIGHT/FULL JOIN: add unmatched right rows
                if (matched_right_bitmap) |bitmap| {
                    for (bitmap, 0..) |matched, right_idx| {
                        if (!matched) {
                            try left_indices.append(self.allocator, std.math.maxInt(usize));
                            try right_indices.append(self.allocator, right_idx);
                        }
                    }
                }
            }
        } else {
            // Slow path: string-based hash table for non-integer keys
            var hash_table = std.StringHashMap(std.ArrayListUnmanaged(usize)).init(self.allocator);
            defer {
                var iter = hash_table.iterator();
                while (iter.next()) |entry| {
                    self.allocator.free(entry.key_ptr.*);
                    entry.value_ptr.deinit(self.allocator);
                }
                hash_table.deinit();
            }

            // Use bitmap for outer join tracking
            var matched_right_str: ?[]bool = null;
            defer if (matched_right_str) |b| self.allocator.free(b);
            if (join_clause.join_type == .right or join_clause.join_type == .full) {
                matched_right_str = try self.allocator.alloc(bool, right_key_data.len());
                @memset(matched_right_str.?, false);
            }

            for (0..right_key_data.len()) |idx| {
                const key = try self.joinKeyToString(right_key_data, idx);
                defer self.allocator.free(key);

                const result = try hash_table.getOrPut(key);
                if (!result.found_existing) {
                    const key_copy = try self.allocator.dupe(u8, key);
                    result.key_ptr.* = key_copy;
                    result.value_ptr.* = .{};
                }
                try result.value_ptr.append(self.allocator, idx);
            }

            for (0..left_key_data.len()) |left_idx| {
                const key = try self.joinKeyToString(left_key_data, left_idx);
                defer self.allocator.free(key);

                if (hash_table.get(key)) |right_list| {
                    for (right_list.items) |right_idx| {
                        try left_indices.append(self.allocator, left_idx);
                        try right_indices.append(self.allocator, right_idx);
                        if (matched_right_str) |b| b[right_idx] = true;
                    }
                } else if (join_clause.join_type == .left or join_clause.join_type == .full) {
                    try left_indices.append(self.allocator, left_idx);
                    try right_indices.append(self.allocator, std.math.maxInt(usize));
                }
            }

            // RIGHT/FULL JOIN: add unmatched right rows
            if (matched_right_str) |bitmap| {
                for (bitmap, 0..) |matched, right_idx| {
                    if (!matched) {
                        try left_indices.append(self.allocator, std.math.maxInt(usize));
                        try right_indices.append(self.allocator, right_idx);
                    }
                }
            }
        }

        const joined_data = try self.allocator.create(JoinedData);
        errdefer self.allocator.destroy(joined_data);

        joined_data.* = JoinedData{
            .columns = std.StringHashMap(CachedColumn).init(self.allocator),
            .column_names = &[_][]const u8{},
            .row_count = left_indices.items.len,
            .allocator = self.allocator,
            .left_table = left_table,
        };
        errdefer joined_data.deinit();

        // Build column names list and copy data
        var col_names = std.ArrayListUnmanaged([]const u8){};
        errdefer {
            for (col_names.items) |name| {
                self.allocator.free(name);
            }
            col_names.deinit(self.allocator);
        }

        // Extract needed column names from SELECT items for projection pushdown
        // Store unqualified names - addJoinedColumnsFiltered handles alias matching
        var needed_cols: ?[]const []const u8 = null;
        var needed_cols_buf: [64][]const u8 = undefined;
        var needed_count: usize = 0;

        if (self.current_select_items) |items| {
            for (items) |item| {
                if (needed_count >= 64) break;
                switch (item.expr) {
                    .column => |col| {
                        // Check for star (*) - means all columns needed
                        if (std.mem.eql(u8, col.name, "*")) {
                            needed_count = 0;
                            break;
                        }
                        // Store just the column name - filter function handles alias matching
                        needed_cols_buf[needed_count] = col.name;
                        needed_count += 1;
                    },
                    else => {},
                }
            }
            if (needed_count > 0) {
                needed_cols = needed_cols_buf[0..needed_count];
            }
        }

        // Add left table columns (with table alias prefix if available)
        const left_alias = switch (left_ref.*) {
            .simple => |s| s.alias orelse s.name,
            else => "left",
        };

        try self.addJoinedColumnsFiltered(
            left_table,
            left_alias,
            left_indices.items,
            joined_data,
            &col_names,
            false, // isRightSide
            needed_cols,
        );

        // Add right table columns
        const right_alias = switch (join_clause.table.*) {
            .simple => |s| s.alias orelse s.name,
            else => "right",
        };

        try self.addJoinedColumnsFiltered(
            right_table,
            right_alias,
            right_indices.items,
            joined_data,
            &col_names,
            true, // isRightSide
            needed_cols,
        );

        joined_data.column_names = try col_names.toOwnedSlice(self.allocator);

        // Release left_source ownership since it's now managed by joined_data
        // (we keep left_table pointer in joined_data.left_table)
        switch (left_source) {
            .direct => {}, // Nothing to release
            .logic_table => |*lt| {
                lt.executor.deinit();
                self.allocator.destroy(lt.executor);
            },
            .joined => |jd| {
                jd.deinit();
                self.allocator.destroy(jd);
            },
        }

        return .{ .joined = joined_data };
    }

    /// Execute a SELECT query on a materialized joined table
    fn executeOnJoinedTable(self: *Self, stmt: *const SelectStmt, params: []const Value, joined_data: *JoinedData) !Result {
        _ = params;

        // Get row count from the joined data
        const row_count = joined_data.row_count;

        // Build result columns from SELECT list
        var result_columns: std.ArrayList(Result.Column) = .{};
        defer result_columns.deinit(self.allocator);

        for (stmt.columns) |item| {
            const col_name = switch (item.expr) {
                .column => |col| blk: {
                    // Handle qualified names: "a.id" -> "a_id"
                    if (col.table) |table_name| {
                        var name_buf: [64]u8 = undefined;
                        const qualified = std.fmt.bufPrint(&name_buf, "{s}_{s}", .{ table_name, col.name }) catch col.name;
                        break :blk qualified;
                    }
                    break :blk col.name;
                },
                else => continue, // Skip non-column expressions for now
            };

            const output_name = item.alias orelse switch (item.expr) {
                .column => |col| col.name,
                else => "expr",
            };

            // Look up the column in joined_data.columns
            const col_data_opt = joined_data.columns.get(col_name) orelse blk: {
                // Try without table qualifier
                const simple_name = switch (item.expr) {
                    .column => |col| col.name,
                    else => break :blk null,
                };
                break :blk joined_data.columns.get(simple_name);
            };

            if (col_data_opt) |col_data| {
                // Convert CachedColumn to Result.ColumnData
                const result_data: Result.ColumnData = switch (col_data) {
                    .int64 => |d| .{ .int64 = try self.allocator.dupe(i64, d) },
                    .int32 => |d| .{ .int32 = try self.allocator.dupe(i32, d) },
                    .float64 => |d| .{ .float64 = try self.allocator.dupe(f64, d) },
                    .float32 => |d| .{ .float32 = try self.allocator.dupe(f32, d) },
                    .bool_ => |d| .{ .bool_ = try self.allocator.dupe(bool, d) },
                    .string => |d| blk: {
                        const duped = try self.allocator.alloc([]const u8, d.len);
                        for (d, 0..) |s, i| {
                            duped[i] = try self.allocator.dupe(u8, s);
                        }
                        break :blk .{ .string = duped };
                    },
                    .timestamp_ns => |d| .{ .timestamp_ns = try self.allocator.dupe(i64, d) },
                    .timestamp_us => |d| .{ .timestamp_us = try self.allocator.dupe(i64, d) },
                    .timestamp_ms => |d| .{ .timestamp_ms = try self.allocator.dupe(i64, d) },
                    .timestamp_s => |d| .{ .timestamp_s = try self.allocator.dupe(i64, d) },
                    .date32 => |d| .{ .date32 = try self.allocator.dupe(i32, d) },
                    .date64 => |d| .{ .date64 = try self.allocator.dupe(i64, d) },
                };
                try result_columns.append(self.allocator, Result.Column{
                    .name = output_name,
                    .data = result_data,
                });
            }
        }

        return Result{
            .columns = try result_columns.toOwnedSlice(self.allocator),
            .row_count = row_count,
            .allocator = self.allocator,
        };
    }

    /// Extract left and right column names from JOIN ON condition
    /// Auto-resolves columns with vector indexes to their shadow columns
    fn extractJoinKeys(self: *Self, condition: ast.Expr, left_table_name: []const u8, right_table_name: []const u8) !struct { left_col: []const u8, right_col: []const u8 } {
        // ON condition should be: left.col = right.col
        switch (condition) {
            .binary => |bin| {
                if (bin.op != .eq) return error.JoinConditionMustBeEquality;

                var left_col = switch (bin.left.*) {
                    .column => |col| col.name, // column.table is optional qualifier
                    else => return error.JoinConditionMustBeColumn,
                };

                var right_col = switch (bin.right.*) {
                    .column => |col| col.name,
                    else => return error.JoinConditionMustBeColumn,
                };

                // Auto-resolve vector index columns to shadow columns
                // If a column has a vector index, use the shadow column for vector operations
                if (self.getVectorIndex(left_table_name, left_col)) |idx_info| {
                    left_col = idx_info.shadow_column;
                }
                if (self.getVectorIndex(right_table_name, right_col)) |idx_info| {
                    right_col = idx_info.shadow_column;
                }

                return .{ .left_col = left_col, .right_col = right_col };
            },
            else => return error.JoinConditionMustBeBinary,
        }
    }

    /// Join key data union for different column types
    const JoinKeyData = union(enum) {
        int64: []i64,
        int32: []i32,
        float64: []f64,
        string: [][]const u8,

        fn len(self: JoinKeyData) usize {
            return switch (self) {
                .int64 => |d| d.len,
                .int32 => |d| d.len,
                .float64 => |d| d.len,
                .string => |d| d.len,
            };
        }
    };

    /// Read join key column data
    fn readJoinKeyColumn(self: *Self, table: *Table, col_idx: u32) !JoinKeyData {
        _ = self;
        const fld = table.getFieldById(col_idx) orelse return error.InvalidColumn;
        const col_type = LanceColumnType.fromLogicalType(fld.logical_type);

        return switch (col_type) {
            .int64, .timestamp_ns, .timestamp_us, .timestamp_ms, .timestamp_s, .date64 => .{ .int64 = try table.readInt64Column(col_idx) },
            .int32, .date32 => .{ .int32 = try table.readInt32Column(col_idx) },
            .float32, .float64 => .{ .float64 = try table.readFloat64Column(col_idx) },
            .string => .{ .string = try table.readStringColumn(col_idx) },
            .bool_, .unsupported => error.UnsupportedJoinKeyType,
        };
    }

    /// Free join key data
    fn freeJoinKeyData(self: *Self, data: JoinKeyData) void {
        switch (data) {
            .int64 => |d| self.allocator.free(d),
            .int32 => |d| self.allocator.free(d),
            .float64 => |d| self.allocator.free(d),
            .string => |d| {
                for (d) |s| self.allocator.free(s);
                self.allocator.free(d);
            },
        }
    }

    /// Convert join key value at index to string for hashing
    fn joinKeyToString(self: *Self, data: JoinKeyData, idx: usize) ![]u8 {
        var buf: [64]u8 = undefined;
        const result = switch (data) {
            .int64 => |d| std.fmt.bufPrint(&buf, "{d}", .{d[idx]}),
            .int32 => |d| std.fmt.bufPrint(&buf, "{d}", .{d[idx]}),
            .float64 => |d| std.fmt.bufPrint(&buf, "{d:.10}", .{d[idx]}),
            .string => |d| return try self.allocator.dupe(u8, d[idx]),
        };
        return try self.allocator.dupe(u8, result catch return error.FormatError);
    }

    /// Add columns from a table to the joined result
    fn addJoinedColumns(
        self: *Self,
        table: *Table,
        alias: []const u8,
        row_indices: []const usize,
        joined_data: *JoinedData,
        col_names: *std.ArrayListUnmanaged([]const u8),
        is_right_side: bool,
    ) !void {
        try self.addJoinedColumnsFiltered(table, alias, row_indices, joined_data, col_names, is_right_side, null);
    }

    /// Add columns from a table to the joined result, optionally filtered to specific columns
    fn addJoinedColumnsFiltered(
        self: *Self,
        table: *Table,
        alias: []const u8,
        row_indices: []const usize,
        joined_data: *JoinedData,
        col_names: *std.ArrayListUnmanaged([]const u8),
        is_right_side: bool,
        needed_cols: ?[]const []const u8, // If non-null, only include these columns
    ) !void {
        const schema = table.getSchema() orelse return error.NoSchema;

        for (schema.fields) |field| {
            if (field.id < 0) continue;
            const col_idx: u32 = @intCast(field.id);

            // Skip unsupported column types (e.g., fixed_size_list for embeddings)
            const col_type = LanceColumnType.fromLogicalType(field.logical_type);
            if (col_type == .unsupported or col_type == .bool_) continue;

            // Projection pushdown: skip columns not in needed list
            if (needed_cols) |cols| {
                var found = false;
                for (cols) |needed| {
                    // Check both unqualified and qualified name
                    if (std.mem.eql(u8, field.name, needed)) {
                        found = true;
                        break;
                    }
                    // Check alias.column format
                    if (std.mem.startsWith(u8, needed, alias)) {
                        const suffix = needed[alias.len..];
                        if (suffix.len > 0 and suffix[0] == '.' and std.mem.eql(u8, suffix[1..], field.name)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) continue;
            }

            // Create qualified column name: "alias.column"
            const qualified_name = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ alias, field.name });
            errdefer self.allocator.free(qualified_name);

            // Read and filter column data based on row indices
            const col_data = self.readJoinedColumnData(table, col_idx, row_indices, is_right_side) catch |e| {
                // Skip columns with unsupported encoding (e.g., dictionary encoding)
                if (e == error.DictionaryEncodingNotSupported) {
                    self.allocator.free(qualified_name);
                    continue;
                }
                return e;
            };

            try joined_data.columns.put(qualified_name, col_data);
            try col_names.append(self.allocator, qualified_name);
        }
    }

    /// Read column data for joined rows (handles NULL for outer joins)
    /// Uses gather API for efficient single-pass extraction
    fn readJoinedColumnData(
        self: *Self,
        table: *Table,
        col_idx: u32,
        row_indices: []const usize,
        is_right_side: bool,
    ) !CachedColumn {
        _ = self;
        _ = is_right_side;
        const fld = table.getFieldById(col_idx) orelse return error.InvalidColumn;
        const col_type = LanceColumnType.fromLogicalType(fld.logical_type);

        return switch (col_type) {
            .int64, .int32, .timestamp_ns, .timestamp_us, .timestamp_ms, .timestamp_s, .date32, .date64 => .{
                .int64 = try table.gatherInt64Column(col_idx, row_indices, 0),
            },
            .float32, .float64 => .{
                .float64 = try table.gatherFloat64Column(col_idx, row_indices, std.math.nan(f64)),
            },
            .string => blk: {
                // String gather still needs manual handling due to allocations
                const null_idx = std.math.maxInt(usize);
                const all_data = try table.readStringColumn(col_idx);
                defer {
                    for (all_data) |s| table.allocator.free(s);
                    table.allocator.free(all_data);
                }
                const result = try table.allocator.alloc([]const u8, row_indices.len);
                for (row_indices, 0..) |idx, i| {
                    result[i] = try table.allocator.dupe(u8, if (idx == null_idx) "" else all_data[idx]);
                }
                break :blk .{ .string = result };
            },
            .bool_, .unsupported => error.UnsupportedColumnType,
        };
    }

    /// Release resources associated with a table source
    fn releaseTableSource(self: *Self, source: *TableSource) void {
        switch (source.*) {
            .direct => {
                // Nothing to release - table is managed externally
            },
            .logic_table => |*lt| {
                // Clean up executor and free heap allocation
                lt.executor.deinit();
                self.allocator.destroy(lt.executor);
            },
            .joined => |jd| {
                // Clean up joined data
                jd.deinit();
                self.allocator.destroy(jd);
            },
        }
    }

    /// Execute a SELECT statement
    pub fn execute(self: *Self, stmt: *const SelectStmt, params: []const Value) !Result {
        const has_typed_table = self.hasTypedTable();

        var source: ?TableSource = null;
        var original_table: ?*Table = null;
        if (!has_typed_table) {
            // Set SELECT items for projection pushdown in JOINs
            self.current_select_items = stmt.columns;
            source = try self.resolveTableSource(&stmt.from);
            self.active_source = source;
            original_table = self.table;
            self.table = source.?.getTable();
        }
        defer {
            if (source) |*s| self.releaseTableSource(s);
            if (!has_typed_table) {
                self.active_source = null;
                self.table = original_table;
                self.current_select_items = null;
            }
        }

        // Execute compiled query
        var result = try self.executeCompiled(stmt, params);

        // Handle set operations (UNION/INTERSECT/EXCEPT) if present
        if (stmt.set_operation) |set_op| {
            result = try self.executeSetOperation(result, set_op, &[_]Value{});
        }

        return result;
    }

    // ========================================================================
    // Streaming Execution (memory-efficient for large datasets)
    // ========================================================================

    /// Execute a SELECT statement using streaming (processes VECTOR_SIZE batches)
    /// This avoids loading entire columns into memory at once.
    /// Returns null if streaming is not applicable (falls back to compiled path)
    pub fn executeStreaming(self: *Self, stmt: *const SelectStmt) !?Result {
        // Only support streaming for Lance tables
        const table = self.table orelse return null;

        // Check if query is streaming-compatible
        if (!self.isStreamingCompatible(stmt)) return null;

        // Create TableColumnReader
        var col_reader = streaming_reader.TableColumnReader.init(table, self.allocator) catch return null;
        defer col_reader.deinit();

        // Detect query type and execute
        const has_group_by = stmt.group_by != null;
        const has_aggs = self.hasAggregates(stmt.columns);

        if (has_group_by) {
            return try self.executeStreamingGroupBy(&col_reader, stmt);
        } else if (has_aggs) {
            return try self.executeStreamingAggregates(&col_reader, stmt);
        } else {
            // Simple SELECT - not yet optimized for streaming
            return null;
        }
    }

    /// Check if query can use streaming execution
    fn isStreamingCompatible(self: *Self, stmt: *const SelectStmt) bool {
        // Currently support:
        // - Simple aggregates (COUNT, SUM, AVG, MIN, MAX)
        // - GROUP BY with single key column
        // Not yet supported:
        // - JOINs
        // - Window functions
        // - Complex expressions in aggregates
        // - Multiple GROUP BY columns

        // No JOINs for now
        if (stmt.from) |from| {
            switch (from) {
                .join => return false,
                else => {},
            }
        }

        // No HAVING clause for now
        if (stmt.having != null) return false;

        // No ORDER BY for now (streaming doesn't preserve order)
        if (stmt.order_by != null) return false;

        // Check aggregates are supported
        for (stmt.columns) |col| {
            switch (col) {
                .expr => |e| {
                    if (!self.isStreamingSupportedExpr(e.expr)) return false;
                },
                .all, .all_from_table => {},
            }
        }

        return true;
    }

    /// Check if expression is supported in streaming mode
    fn isStreamingSupportedExpr(self: *Self, expr: *const ast.Expr) bool {
        _ = self;
        switch (expr.*) {
            .column => return true,
            .literal => return true,
            .function => |f| {
                // Only support basic aggregates
                if (aggregate_functions.isAggregateFunction(f.name)) {
                    const agg_type = aggregate_functions.parseAggregateType(f.name);
                    if (agg_type) |t| {
                        return switch (t) {
                            .count, .count_star, .sum, .avg, .min, .max => true,
                            else => false, // stddev, variance, median, percentile not yet supported
                        };
                    }
                }
                return false; // Non-aggregate functions not supported in streaming
            },
            .binary => |b| {
                // Allow simple arithmetic in WHERE but not in SELECT
                _ = b;
                return false;
            },
            else => return false,
        }
    }

    /// Execute streaming aggregates (no GROUP BY)
    fn executeStreamingAggregates(
        self: *Self,
        col_reader: *streaming_reader.TableColumnReader,
        stmt: *const SelectStmt,
    ) !Result {
        // Build AggSpec array
        var agg_specs = std.ArrayList(vector_engine.AggSpec).init(self.allocator);
        defer agg_specs.deinit();

        var col_names = std.ArrayList([]const u8).init(self.allocator);
        defer col_names.deinit();

        for (stmt.columns) |col| {
            switch (col) {
                .expr => |e| {
                    const spec = try self.exprToAggSpec(col_reader, e.expr, e.alias);
                    if (spec) |s| {
                        try agg_specs.append(s);
                        try col_names.append(e.alias orelse s.output_name);
                    }
                },
                else => {},
            }
        }

        if (agg_specs.items.len == 0) return error.NoAggregates;

        // Create StreamingExecutor
        var reader_iface = col_reader.reader();
        var executor = try vector_engine.StreamingExecutor.init(
            self.allocator,
            agg_specs.items.len,
        );
        defer executor.deinit();

        // Build filter condition if WHERE clause exists
        const filter = try self.buildStreamingFilter(col_reader, stmt.where_clause);

        // Execute
        const results = try executor.executeAggregate(
            &reader_iface,
            filter,
            agg_specs.items,
        );
        defer self.allocator.free(results);

        // Convert to Result
        return self.aggResultsToResult(results, col_names.items);
    }

    /// Execute streaming GROUP BY
    fn executeStreamingGroupBy(
        self: *Self,
        col_reader: *streaming_reader.TableColumnReader,
        stmt: *const SelectStmt,
    ) !Result {
        const group_by = stmt.group_by orelse return error.NoGroupBy;

        // Get the key column index
        if (group_by.exprs.len != 1) return error.MultipleGroupByNotSupported;
        const key_expr = group_by.exprs[0];
        const key_col_name = switch (key_expr.*) {
            .column => |c| c.name,
            else => return error.ComplexGroupByNotSupported,
        };
        const key_col_idx = col_reader.getColumnIndex(key_col_name) orelse return error.ColumnNotFound;

        // Build AggSpec array
        var agg_specs = std.ArrayList(vector_engine.AggSpec).init(self.allocator);
        defer agg_specs.deinit();

        var col_names = std.ArrayList([]const u8).init(self.allocator);
        defer col_names.deinit();

        // Add key column name first
        try col_names.append(key_col_name);

        for (stmt.columns) |col| {
            switch (col) {
                .expr => |e| {
                    // Skip the key column (it's not an aggregate)
                    if (e.expr.* == .column) {
                        if (std.mem.eql(u8, e.expr.column.name, key_col_name)) continue;
                    }
                    const spec = try self.exprToAggSpec(col_reader, e.expr, e.alias);
                    if (spec) |s| {
                        try agg_specs.append(s);
                        try col_names.append(e.alias orelse s.output_name);
                    }
                },
                else => {},
            }
        }

        // Create StreamingExecutor
        var reader_iface = col_reader.reader();
        var executor = try vector_engine.StreamingExecutor.init(
            self.allocator,
            agg_specs.items.len + 1, // +1 for key column
        );
        defer executor.deinit();

        // Build filter condition
        const filter = try self.buildStreamingFilter(col_reader, stmt.where_clause);

        // Execute GROUP BY
        const gb_result = try executor.executeGroupBy(
            &reader_iface,
            key_col_idx,
            filter,
            agg_specs.items,
        );
        defer {
            self.allocator.free(gb_result.keys);
            for (gb_result.results) |r| self.allocator.free(r);
            self.allocator.free(gb_result.results);
        }

        // Convert to Result
        return self.groupByResultToResult(gb_result, col_names.items);
    }

    /// Convert AST expression to AggSpec
    fn exprToAggSpec(
        self: *Self,
        col_reader: *streaming_reader.TableColumnReader,
        expr: *const ast.Expr,
        alias: ?[]const u8,
    ) !?vector_engine.AggSpec {
        switch (expr.*) {
            .function => |f| {
                const agg_type = aggregate_functions.parseAggregateTypeWithArgs(f.name, f.args);

                // Map to ve.AggFunc
                const func: vector_engine.AggFunc = switch (agg_type) {
                    .count, .count_star => .count,
                    .sum => .sum,
                    .avg => .avg,
                    .min => .min,
                    .max => .max,
                    else => return null, // Unsupported aggregate
                };

                // Get column index (COUNT(*) uses 0)
                var col_idx: usize = 0;
                if (agg_type != .count_star and f.args.len > 0) {
                    const arg = f.args[0];
                    if (arg == .column) {
                        col_idx = col_reader.getColumnIndex(arg.column.name) orelse return error.ColumnNotFound;
                    }
                }

                // Generate output name
                const output_name = alias orelse f.name;

                return vector_engine.AggSpec{
                    .func = func,
                    .col_idx = col_idx,
                    .output_name = output_name,
                };
            },
            else => return null,
        }
        _ = self;
    }

    /// Build filter condition from WHERE clause
    fn buildStreamingFilter(
        self: *Self,
        col_reader: *streaming_reader.TableColumnReader,
        where_clause: ?*const ast.Expr,
    ) !?vector_engine.FilterCondition {
        _ = self;
        const where = where_clause orelse return null;

        // Only support simple comparisons for now: column op literal
        switch (where.*) {
            .binary => |b| {
                // Get column
                const col_name = switch (b.left.*) {
                    .column => |c| c.name,
                    else => return null,
                };
                const col_idx = col_reader.getColumnIndex(col_name) orelse return null;

                // Get operator
                const op: vector_engine.FilterOp = switch (b.op) {
                    .eq => .eq,
                    .ne => .ne,
                    .lt => .lt,
                    .le => .le,
                    .gt => .gt,
                    .ge => .ge,
                    else => return null,
                };

                // Get value
                const value: vector_engine.FilterValue = switch (b.right.*) {
                    .literal => |lit| switch (lit) {
                        .integer => |i| .{ .int64 = i },
                        .float => |f| .{ .float64 = f },
                        else => return null,
                    },
                    else => return null,
                };

                return vector_engine.FilterCondition{
                    .col_idx = col_idx,
                    .op = op,
                    .value = value,
                };
            },
            else => return null,
        }
    }

    /// Convert aggregate results to Result
    fn aggResultsToResult(self: *Self, results: []f64, col_names: []const []const u8) !Result {
        var columns = try self.allocator.alloc(Result.Column, results.len);
        errdefer self.allocator.free(columns);

        for (results, 0..) |val, i| {
            const data = try self.allocator.alloc(f64, 1);
            data[0] = val;
            columns[i] = .{
                .name = col_names[i],
                .data = .{ .float64 = data },
            };
        }

        return Result{
            .columns = columns,
            .row_count = 1,
            .allocator = self.allocator,
            .owns_data = true,
        };
    }

    /// Convert GROUP BY results to Result
    fn groupByResultToResult(
        self: *Self,
        gb_result: struct { keys: []i64, results: [][]f64 },
        col_names: []const []const u8,
    ) !Result {
        const num_groups = gb_result.keys.len;
        const num_aggs = if (gb_result.results.len > 0) gb_result.results[0].len else 0;

        var columns = try self.allocator.alloc(Result.Column, 1 + num_aggs);
        errdefer self.allocator.free(columns);

        // Key column
        const key_data = try self.allocator.alloc(i64, num_groups);
        @memcpy(key_data, gb_result.keys);
        columns[0] = .{
            .name = col_names[0],
            .data = .{ .int64 = key_data },
        };

        // Aggregate columns
        for (0..num_aggs) |a| {
            const agg_data = try self.allocator.alloc(f64, num_groups);
            for (0..num_groups) |g| {
                agg_data[g] = gb_result.results[g][a];
            }
            columns[1 + a] = .{
                .name = col_names[1 + a],
                .data = .{ .float64 = agg_data },
            };
        }

        return Result{
            .columns = columns,
            .row_count = num_groups,
            .allocator = self.allocator,
            .owns_data = true,
        };
    }

    // ========================================================================
    // Statement Execution (handles all statement types)
    // ========================================================================

    /// Result type for non-SELECT statements
    /// Diff result - shows added and deleted rows between versions
    pub const DiffResult = struct {
        /// Added rows (in to_version but not in from_version)
        added: Result,
        /// Deleted rows (in from_version but not in to_version)
        deleted: Result,
        /// From version number (resolved)
        from_version: u32,
        /// To version number (resolved)
        to_version: u32,
        /// Number of fragments added
        fragments_added: u32 = 0,
        /// Number of fragments deleted
        fragments_deleted: u32 = 0,
        /// Total rows in added fragments
        rows_added: u64 = 0,
        /// Total rows in deleted fragments
        rows_deleted: u64 = 0,
    };

    /// Version info for SHOW VERSIONS result
    pub const VersionInfo = struct {
        version: u32,
        timestamp: i64, // Unix timestamp (milliseconds)
        timestamp_str: []const u8 = "", // ISO 8601 formatted timestamp
        operation: []const u8, // INSERT, DELETE, UPDATE
        row_count: u64, // Total rows in this version
        delta: []const u8, // "+3", "-1", etc.
    };

    pub const DropVectorIndexResult = struct { table: []const u8, column: []const u8 };

    pub const StatementResult = union(enum) {
        select: Result,
        vector_index_created: VectorIndexInfo,
        vector_index_dropped: DropVectorIndexResult,
        vector_indexes_list: []const VectorIndexInfo,
        diff_result: DiffResult,
        versions_list: []const VersionInfo,
        changes_list: Result, // Uses standard Result with change column
        no_op: void,
    };

    /// Execute any SQL statement (SELECT, CREATE VECTOR INDEX, etc.)
    pub fn executeStatement(self: *Self, stmt: *const ast.Statement, params: []const Value) !StatementResult {
        return switch (stmt.*) {
            .select => |*select| StatementResult{ .select = try self.execute(select, params) },
            .create_vector_index => |vi| StatementResult{ .vector_index_created = try self.executeCreateVectorIndex(&vi) },
            .drop_vector_index => |vi| StatementResult{ .vector_index_dropped = try self.executeDropVectorIndex(&vi) },
            .show_vector_indexes => |vi| StatementResult{ .vector_indexes_list = try self.executeShowVectorIndexes(&vi) },
            .diff => |d| StatementResult{ .diff_result = try self.executeDiff(&d) },
            .show_versions => |sv| StatementResult{ .versions_list = try self.executeShowVersions(&sv) },
            .show_changes => |sc| StatementResult{ .changes_list = try self.executeShowChanges(&sc) },
        };
    }

    // ========================================================================
    // Vector Index Execution
    // ========================================================================

    /// Default dimensions for supported embedding models
    fn getModelDimension(model: []const u8) u32 {
        // Case-insensitive comparison
        var upper_buf: [32]u8 = undefined;
        const model_len = @min(model.len, upper_buf.len);
        for (model[0..model_len], 0..) |ch, i| {
            upper_buf[i] = std.ascii.toUpper(ch);
        }
        const upper = upper_buf[0..model_len];

        if (std.mem.eql(u8, upper, "MINILM")) return 384;
        if (std.mem.eql(u8, upper, "CLIP")) return 512;
        if (std.mem.eql(u8, upper, "BGE-SMALL")) return 384;
        if (std.mem.eql(u8, upper, "BGE-BASE")) return 768;
        if (std.mem.eql(u8, upper, "BGE-LARGE")) return 1024;
        if (std.mem.eql(u8, upper, "OPENAI")) return 1536;
        if (std.mem.eql(u8, upper, "COHERE")) return 1024;

        // Default to MiniLM dimension
        return 384;
    }

    /// Execute CREATE VECTOR INDEX statement
    fn executeCreateVectorIndex(self: *Self, stmt: *const ast.CreateVectorIndexStmt) !VectorIndexInfo {
        // Build the index key: "table.column"
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ stmt.table_name, stmt.column_name }) catch return error.KeyTooLong;

        // Check if index already exists
        if (self.vector_indexes.contains(key)) {
            if (stmt.if_not_exists) {
                // Return existing index info
                return self.vector_indexes.get(key).?;
            }
            return error.VectorIndexAlreadyExists;
        }

        // Determine dimension
        const dimension = stmt.dimension orelse getModelDimension(stmt.model);

        // Build shadow column name: "__vec_{column}_{model}"
        var shadow_buf: [128]u8 = undefined;
        const shadow_column = std.fmt.bufPrint(&shadow_buf, "__vec_{s}_{s}", .{ stmt.column_name, stmt.model }) catch return error.ShadowColumnNameTooLong;

        // Allocate strings for persistent storage
        const alloc_key = try self.allocator.dupe(u8, key);
        const alloc_shadow = try self.allocator.dupe(u8, shadow_column);

        const info = VectorIndexInfo{
            .table_name = stmt.table_name,
            .column_name = stmt.column_name,
            .model = stmt.model,
            .shadow_column = alloc_shadow,
            .dimension = dimension,
            .created_at = std.time.timestamp(),
        };

        try self.vector_indexes.put(alloc_key, info);

        return info;
    }

    /// Execute DROP VECTOR INDEX statement
    fn executeDropVectorIndex(self: *Self, stmt: *const ast.DropVectorIndexStmt) !DropVectorIndexResult {
        // Build the index key: "table.column"
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ stmt.table_name, stmt.column_name }) catch return error.KeyTooLong;

        // Check if index exists
        if (!self.vector_indexes.contains(key)) {
            if (stmt.if_exists) {
                return .{ .table = stmt.table_name, .column = stmt.column_name };
            }
            return error.VectorIndexNotFound;
        }

        // Remove the index
        const kv = self.vector_indexes.fetchRemove(key);
        if (kv) |entry| {
            // Free allocated strings
            self.allocator.free(entry.key);
            self.allocator.free(entry.value.shadow_column);
        }

        return .{ .table = stmt.table_name, .column = stmt.column_name };
    }

    /// Execute SHOW VECTOR INDEXES statement
    fn executeShowVectorIndexes(self: *Self, stmt: *const ast.ShowVectorIndexesStmt) ![]const VectorIndexInfo {
        var results = std.ArrayList(VectorIndexInfo){};
        errdefer results.deinit(self.allocator);

        var it = self.vector_indexes.iterator();
        while (it.next()) |entry| {
            const info = entry.value_ptr.*;

            // Filter by table name if specified
            if (stmt.table_name) |table| {
                if (!std.mem.eql(u8, info.table_name, table)) {
                    continue;
                }
            }

            try results.append(self.allocator, info);
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Get vector index info for a table.column
    pub fn getVectorIndex(self: *Self, table_name: []const u8, column_name: []const u8) ?VectorIndexInfo {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ table_name, column_name }) catch return null;
        return self.vector_indexes.get(key);
    }

    /// Check if a column has a vector index
    pub fn hasVectorIndex(self: *Self, table_name: []const u8, column_name: []const u8) bool {
        return self.getVectorIndex(table_name, column_name) != null;
    }

    // ========================================================================
    // Time Travel / Diff Execution
    // ========================================================================

    /// Resolve a VersionRef to an absolute version number
    fn resolveVersion(self: *Self, ref: ast.VersionRef, table_ref: *const ast.TableRef) !u32 {
        _ = table_ref; // Will be used when we load manifest to get current version

        // Get current version from table (if available)
        const current_version: u32 = if (self.table) |t|
            t.lance_file.footer.major_version
        else
            1;

        return switch (ref) {
            .absolute => |v| v,
            .relative => |offset| blk: {
                // -1 means HEAD~1, -2 means HEAD~2
                const abs_offset: u32 = @intCast(if (offset < 0) -offset else offset);
                if (current_version < abs_offset) return error.VersionBeforeStart;
                break :blk current_version - abs_offset;
            },
            .head => current_version,
            .head_offset => |offset| blk: {
                if (current_version < offset) return error.VersionBeforeStart;
                break :blk current_version - offset;
            },
            .current => current_version,
        };
    }

    /// Execute DIFF statement - compare two versions
    fn executeDiff(self: *Self, stmt: *const ast.DiffStmt) !DiffResult {
        // Resolve version numbers
        const from_version = try self.resolveVersion(stmt.from_version, &stmt.table_ref);
        const to_version = if (stmt.to_version) |tv|
            try self.resolveVersion(tv, &stmt.table_ref)
        else if (self.table) |t|
            t.lance_file.footer.major_version
        else
            1;

        // Get dataset path
        const dataset_path = self.dataset_path orelse {
            // Try to get from table_ref if it's a function call
            switch (stmt.table_ref) {
                .function => |func| {
                    if (func.func.args.len > 0) {
                        switch (func.func.args[0]) {
                            .value => |val| switch (val) {
                                .string => |s| return self.computeDiff(s, from_version, to_version, stmt.limit),
                                else => {},
                            },
                            else => {},
                        }
                    }
                },
                else => {},
            }
            // No path - return empty diff
            return self.emptyDiffResult(from_version, to_version);
        };

        return self.computeDiff(dataset_path, from_version, to_version, stmt.limit);
    }

    /// Create an empty DiffResult
    fn emptyDiffResult(self: *Self, from_version: u32, to_version: u32) !DiffResult {
        return DiffResult{
            .added = Result{
                .columns = try self.allocator.alloc(Result.Column, 0),
                .row_count = 0,
                .allocator = self.allocator,
                .owns_data = true,
            },
            .deleted = Result{
                .columns = try self.allocator.alloc(Result.Column, 0),
                .row_count = 0,
                .allocator = self.allocator,
                .owns_data = true,
            },
            .from_version = from_version,
            .to_version = to_version,
        };
    }

    /// Compute diff between two versions
    fn computeDiff(self: *Self, dataset_path: []const u8, from_version: u32, to_version: u32, limit: u32) !DiffResult {
        // Load both manifests
        var from_manifest = format.manifest.loadManifest(self.allocator, dataset_path, from_version) catch {
            return self.emptyDiffResult(from_version, to_version);
        };
        defer from_manifest.deinit();

        var to_manifest = format.manifest.loadManifest(self.allocator, dataset_path, to_version) catch {
            return self.emptyDiffResult(from_version, to_version);
        };
        defer to_manifest.deinit();

        // Build sets of fragment paths for comparison
        var from_fragments = std.StringHashMap(u64).init(self.allocator);
        defer from_fragments.deinit();
        for (from_manifest.fragments) |frag| {
            try from_fragments.put(frag.file_path, frag.physical_rows);
        }

        var to_fragments = std.StringHashMap(u64).init(self.allocator);
        defer to_fragments.deinit();
        for (to_manifest.fragments) |frag| {
            try to_fragments.put(frag.file_path, frag.physical_rows);
        }

        // Collect added fragment paths (in to but not in from)
        var added_paths: std.ArrayListUnmanaged([]const u8) = .empty;
        defer added_paths.deinit(self.allocator);
        var added_rows: u64 = 0;
        var to_iter = to_fragments.iterator();
        while (to_iter.next()) |entry| {
            if (!from_fragments.contains(entry.key_ptr.*)) {
                try added_paths.append(self.allocator, entry.key_ptr.*);
                added_rows += entry.value_ptr.*;
            }
        }

        // Collect deleted fragment paths (in from but not in to)
        var deleted_paths: std.ArrayListUnmanaged([]const u8) = .empty;
        defer deleted_paths.deinit(self.allocator);
        var deleted_rows: u64 = 0;
        var from_iter = from_fragments.iterator();
        while (from_iter.next()) |entry| {
            if (!to_fragments.contains(entry.key_ptr.*)) {
                try deleted_paths.append(self.allocator, entry.key_ptr.*);
                deleted_rows += entry.value_ptr.*;
            }
        }

        // Load actual row data from fragments
        const added_result = try self.loadFragmentData(dataset_path, added_paths.items, limit, "ADD");
        const deleted_result = try self.loadFragmentData(dataset_path, deleted_paths.items, limit, "DELETE");

        return DiffResult{
            .added = added_result,
            .deleted = deleted_result,
            .from_version = from_version,
            .to_version = to_version,
            .fragments_added = @intCast(added_paths.items.len),
            .fragments_deleted = @intCast(deleted_paths.items.len),
            .rows_added = added_rows,
            .rows_deleted = deleted_rows,
        };
    }

    /// Load row data from fragment files
    fn loadFragmentData(self: *Self, dataset_path: []const u8, fragment_paths: []const []const u8, limit: u32, change_type: []const u8) !Result {
        if (fragment_paths.len == 0) {
            return Result{
                .columns = try self.allocator.alloc(Result.Column, 0),
                .row_count = 0,
                .allocator = self.allocator,
                .owns_data = true,
            };
        }

        // Collect all rows from all fragments
        var all_rows: std.ArrayListUnmanaged([]Result.Column) = .empty;
        defer {
            for (all_rows.items) |row| {
                self.allocator.free(row);
            }
            all_rows.deinit(self.allocator);
        }

        var schema_columns: ?[][]const u8 = null;
        var total_rows: usize = 0;

        for (fragment_paths) |frag_path| {
            if (total_rows >= limit) break;

            // Build full path: dataset_path/data/fragment_path
            const full_path = std.fs.path.join(self.allocator, &.{ dataset_path, "data", frag_path }) catch continue;
            defer self.allocator.free(full_path);

            // Read fragment file
            const file_data = std.fs.cwd().readFileAlloc(self.allocator, full_path, 100 * 1024 * 1024) catch continue;
            defer self.allocator.free(file_data);

            // Parse as Table
            var table = Table.init(self.allocator, file_data) catch continue;
            defer table.deinit();

            // Get schema on first fragment
            if (schema_columns == null) {
                schema_columns = table.columnNames() catch continue;
            }

            // Get row count for this fragment
            const frag_row_count = table.rowCount(0) catch continue;
            const rows_to_read = @min(frag_row_count, limit - total_rows);

            total_rows += rows_to_read;
        }

        // If we couldn't read any fragments, return empty result
        if (schema_columns == null or total_rows == 0) {
            return Result{
                .columns = try self.allocator.alloc(Result.Column, 0),
                .row_count = 0,
                .allocator = self.allocator,
                .owns_data = true,
            };
        }

        // Now actually load the data
        const col_names = schema_columns.?;
        defer self.allocator.free(col_names);

        // Create columns: change_type + original columns
        const num_cols = col_names.len + 1;
        var columns = try self.allocator.alloc(Result.Column, num_cols);

        // First column is change_type
        const change_types = try self.allocator.alloc([]const u8, total_rows);
        for (change_types) |*ct| {
            ct.* = change_type;
        }
        columns[0] = Result.Column{
            .name = "change",
            .data = .{ .string = change_types },
        };

        // Initialize other columns based on first fragment's schema
        // For simplicity, we'll read all as int64 (can be extended for other types)
        for (col_names, 0..) |col_name, i| {
            const int_data = try self.allocator.alloc(i64, total_rows);
            @memset(int_data, 0); // Initialize to zero
            columns[i + 1] = Result.Column{
                .name = col_name,
                .data = .{ .int64 = int_data },
            };
        }

        // Now read actual data from fragments
        var row_offset: usize = 0;
        for (fragment_paths) |frag_path| {
            if (row_offset >= total_rows) break;

            const full_path = std.fs.path.join(self.allocator, &.{ dataset_path, "data", frag_path }) catch continue;
            defer self.allocator.free(full_path);

            const file_data = std.fs.cwd().readFileAlloc(self.allocator, full_path, 100 * 1024 * 1024) catch continue;
            defer self.allocator.free(file_data);

            var table = Table.init(self.allocator, file_data) catch continue;
            defer table.deinit();

            const frag_row_count = table.rowCount(0) catch continue;
            const rows_to_read = @min(frag_row_count, total_rows - row_offset);

            // Read each column
            for (col_names, 0..) |_, col_idx| {
                const col_data = table.readInt64Column(@intCast(col_idx)) catch continue;
                defer self.allocator.free(col_data);

                const dest = columns[col_idx + 1].data.int64;
                const copy_len = @min(col_data.len, rows_to_read);
                @memcpy(dest[row_offset .. row_offset + copy_len], col_data[0..copy_len]);
            }

            row_offset += rows_to_read;
        }

        return Result{
            .columns = columns,
            .row_count = total_rows,
            .allocator = self.allocator,
            .owns_data = true,
        };
    }

    /// Execute SHOW VERSIONS - list version history with deltas
    fn executeShowVersions(self: *Self, stmt: *const ast.ShowVersionsStmt) ![]const VersionInfo {
        var results = std.ArrayList(VersionInfo){};
        errdefer results.deinit(self.allocator);

        // Get dataset path from executor or try to extract from table_ref
        const dataset_path = self.dataset_path orelse {
            // Fall back to table name if it looks like a path
            switch (stmt.table_ref) {
                .function => |func| {
                    if (func.func.args.len > 0) {
                        switch (func.func.args[0]) {
                            .value => |val| switch (val) {
                                .string => |s| return self.showVersionsFromPath(s, stmt.limit, &results),
                                else => {},
                            },
                            else => {},
                        }
                    }
                },
                else => {},
            }
            // No path available - return empty results
            return results.toOwnedSlice(self.allocator);
        };

        return self.showVersionsFromPath(dataset_path, stmt.limit, &results);
    }

    /// Helper to load version history from a dataset path
    fn showVersionsFromPath(self: *Self, dataset_path: []const u8, limit: ?u32, results: *std.ArrayList(VersionInfo)) ![]const VersionInfo {
        // List all versions in the dataset
        const versions = format.manifest.listVersions(self.allocator, dataset_path) catch {
            // If we can't list versions (e.g., no _versions dir), return empty results
            return results.toOwnedSlice(self.allocator);
        };
        defer self.allocator.free(versions);

        // Apply limit (descending order - newest first)
        const max_versions = limit orelse @as(u32, @intCast(versions.len));

        // Process versions in reverse order (newest first) up to limit
        var count: u32 = 0;
        var i: usize = versions.len;
        while (i > 0 and count < max_versions) : (count += 1) {
            i -= 1;
            const v = versions[i];

            // Load manifest to get row count and timestamp
            var manifest = format.manifest.loadManifest(self.allocator, dataset_path, v) catch continue;
            defer manifest.deinit();

            // Load previous version to calculate delta
            const prev_row_count: u64 = if (i > 0) blk: {
                var prev_manifest = format.manifest.loadManifest(self.allocator, dataset_path, versions[i - 1]) catch break :blk 0;
                defer prev_manifest.deinit();
                break :blk prev_manifest.total_rows;
            } else 0;

            // Calculate delta (this version's rows minus previous version's rows)
            const current_rows = manifest.total_rows;
            const diff = @as(i64, @intCast(current_rows)) - @as(i64, @intCast(prev_row_count));

            const delta_str = blk: {
                var buf: [32]u8 = undefined;
                const len = if (diff >= 0)
                    std.fmt.bufPrint(&buf, "+{d}", .{@as(u64, @intCast(diff))}) catch break :blk "+0"
                else
                    std.fmt.bufPrint(&buf, "{d}", .{diff}) catch break :blk "-0";
                break :blk self.allocator.dupe(u8, len) catch break :blk "+0";
            };

            // Determine operation type based on delta
            const operation = if (diff > 0) "INSERT" else if (diff < 0) "DELETE" else "COMPACT";

            // Format timestamp
            const timestamp_str = manifest.timestamp.format(self.allocator) catch "unknown";

            try results.append(self.allocator, VersionInfo{
                .version = @intCast(v),
                .timestamp_str = timestamp_str,
                .timestamp = manifest.timestamp.toMillis(),
                .operation = operation,
                .row_count = manifest.total_rows,
                .delta = delta_str,
            });
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Execute SHOW CHANGES - list changes since a version
    fn executeShowChanges(self: *Self, stmt: *const ast.ShowChangesStmt) !Result {
        // Resolve the since version
        const since_version = try self.resolveVersion(stmt.since_version, &stmt.table_ref);
        _ = since_version;

        // Return empty result as placeholder
        // Full implementation requires walking version chain

        return Result{
            .columns = try self.allocator.alloc(Result.Column, 0),
            .row_count = 0,
            .allocator = self.allocator,
            .owns_data = true,
        };
    }

    // ========================================================================
    // Set Operation Execution (UNION/INTERSECT/EXCEPT)
    // ========================================================================

    /// Execute a set operation (UNION, INTERSECT, EXCEPT) between two result sets
    /// Note: Takes ownership of left result and frees it after use
    fn executeSetOperation(self: *Self, left_in: Result, set_op_def: ast.SetOperation, params: []const Value) anyerror!Result {
        var left = left_in;
        defer left.deinit();

        // Execute the right-hand SELECT
        var right = try self.execute(set_op_def.right, params);
        defer right.deinit();

        // Verify column count matches
        if (left.columns.len != right.columns.len) {
            return error.SetOperationColumnMismatch;
        }

        return switch (set_op_def.op_type) {
            .union_all => try set_ops.executeUnionAll(self.allocator, left, right),
            .union_distinct => try set_ops.executeUnionDistinct(self.allocator, left, right),
            .intersect => try set_ops.executeIntersect(self.allocator, left, right),
            .except => try set_ops.executeExcept(self.allocator, left, right),
        };
    }

    // ========================================================================
    // GROUP BY / Aggregate Execution
    // ========================================================================

    /// Check if SELECT list contains any aggregate functions
    fn hasAggregates(self: *Self, select_list: []const ast.SelectItem) bool {
        _ = self;
        return aggregate_functions.hasAggregates(select_list);
    }

    /// Recursively check if expression contains an aggregate function
    fn containsAggregate(expr: *const Expr) bool {
        return aggregate_functions.containsAggregate(expr);
    }

    /// Check if function name is an aggregate function
    fn isAggregateFunction(name: []const u8) bool {
        return aggregate_functions.isAggregateFunction(name);
    }

    // ========================================================================
    // Window Function Support
    // ========================================================================

    /// Window function types (re-exported from window_functions module)
    const WindowFunctionType = window_functions.WindowFunctionType;

    /// Check if expression is a window function (has OVER clause)
    fn isWindowFunction(expr: *const Expr) bool {
        return window_functions.isWindowFunction(expr);
    }

    /// Check if SELECT list contains any window functions
    fn hasWindowFunctions(self: *Self, select_list: []const ast.SelectItem) bool {
        _ = self;
        return window_functions.hasWindowFunctions(select_list);
    }

    /// Parse window function name
    fn parseWindowFunctionType(name: []const u8) ?WindowFunctionType {
        return window_functions.parseWindowFunctionType(name);
    }

    /// Evaluate window functions and add result columns
    /// Window functions are evaluated after all base columns are computed
    fn evaluateWindowFunctions(
        self: *Self,
        columns: *std.ArrayList(Result.Column),
        select_list: []const ast.SelectItem,
        indices: []const u32,
    ) !void {
        for (select_list) |item| {
            if (!isWindowFunction(&item.expr)) continue;

            const call = item.expr.call;
            const window_spec = call.window.?;

            // Preload columns needed for window function
            try self.preloadWindowColumns(window_spec, call.args);

            // Create window context
            const ctx = window_eval.WindowContext{
                .allocator = self.allocator,
                .column_cache = &self.column_cache,
            };

            // Evaluate window function
            const results = try window_eval.evaluateWindowFunction(ctx, call, window_spec, indices);

            // Add result column
            const col_name = item.alias orelse call.name;
            try columns.append(self.allocator, Result.Column{
                .name = col_name,
                .data = Result.ColumnData{ .int64 = results },
            });
        }
    }

    /// Preload columns needed for window function evaluation
    fn preloadWindowColumns(self: *Self, window_spec: *const ast.WindowSpec, args: []const Expr) !void {
        var col_names = std.ArrayList([]const u8){};
        defer col_names.deinit(self.allocator);

        // Add PARTITION BY columns
        if (window_spec.partition_by) |partition_cols| {
            for (partition_cols) |col| {
                try col_names.append(self.allocator, col);
            }
        }

        // Add ORDER BY columns
        if (window_spec.order_by) |order_by| {
            for (order_by) |ob| {
                try col_names.append(self.allocator, ob.column);
            }
        }

        // Add LAG/LEAD source column
        if (args.len > 0) {
            if (args[0] == .column) {
                try col_names.append(self.allocator, args[0].column.name);
            }
        }

        try self.preloadColumns(col_names.items);
    }

    /// Parse aggregate function name to AggregateType
    fn parseAggregateType(name: []const u8, args: []const Expr) AggregateType {
        return aggregate_functions.parseAggregateTypeWithArgs(name, args);
    }


    /// Extract column names from any expression (delegates to group_eval)
    fn extractExprColumnNames(self: *Self, expr: *const Expr, list: *std.ArrayList([]const u8)) anyerror!void {
        return group_eval.extractExprColumnNames(self.allocator, expr, list);
    }

    // ========================================================================
    // WHERE Clause Evaluation (delegated to where_eval module)
    // ========================================================================

    /// Static wrapper for evaluateToValue - used as callback from where_eval
    fn evalExprCallback(ctx: *anyopaque, expr: *const Expr, row_idx: u32) anyerror!Value {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.evaluateToValue(expr, row_idx);
    }

    /// Static wrapper for execute - used as callback from where_eval for subqueries
    fn executeCallback(ctx: *anyopaque, stmt: *ast.SelectStmt, params: []const Value) anyerror!Result {
        const self: *Self = @ptrCast(@alignCast(ctx));
        return self.execute(stmt, params);
    }

    /// Evaluate WHERE clause and return matching row indices
    fn evaluateWhere(self: *Self, where_expr: *const Expr, params: []const Value) ![]u32 {
        const row_count = try self.getRowCount();

        const ctx = where_eval.WhereContext{
            .allocator = self.allocator,
            .column_cache = &self.column_cache,
            .row_count = row_count,
            .execute_fn = executeCallback,
            .eval_ctx = self,
            .eval_expr_fn = evalExprCallback,
        };

        return where_eval.evaluateWhere(ctx, where_expr, params);
    }

    /// Evaluate expression to a concrete value (delegates to expr_eval)
    fn evaluateToValue(self: *Self, expr: *const Expr, row_idx: u32) !Value {
        const ctx = self.buildExprContext();
        return expr_eval.evaluateExprToValue(ctx, expr, row_idx);
    }


    // ========================================================================
    // Method Call Evaluation (for @logic_table)
    // ========================================================================

    /// Evaluate a @logic_table method call (e.g., t.risk_score())
    ///
    /// This uses batch dispatch to compute all method results at once, then caches them.
    /// On subsequent calls with different row indices, the cached results are returned.
    ///
    /// For methods that require column data (Phase 3/4), the inputs need to be populated
    /// with ColumnBinding data from the Lance table.
    ///
    fn evaluateMethodCall(self: *Self, mc: anytype, row_idx: u32) !Value {
        // Get class name from alias
        const class_name = self.logic_table_aliases.get(mc.object) orelse
            return error.TableAliasNotFound;

        // Get dispatcher
        var dispatcher = self.dispatcher orelse
            return error.NoDispatcherConfigured;

        // For now, we only support methods with no runtime arguments
        // The compiled method operates on batch data loaded in the LogicTableContext
        if (mc.args.len > 0) {
            return error.MethodArgsNotSupported;
        }

        // Build cache key: "ClassName.methodName"
        var cache_key_buf: [256]u8 = undefined;
        const cache_key = std.fmt.bufPrint(&cache_key_buf, "{s}.{s}", .{ class_name, mc.method }) catch
            return error.CacheKeyTooLong;

        // Check if we have cached results
        if (self.method_results_cache.get(cache_key)) |cached_results| {
            if (row_idx < cached_results.len) {
                return Value{ .float = cached_results[row_idx] };
            }
            return error.RowIndexOutOfBounds;
        }

        // No cached results - compute batch results

        // Determine row count from the current table
        const row_count: usize = blk: {
            if (self.table) |t| {
                // Get row count from column 0
                const count = t.rowCount(0) catch 1000;
                break :blk @intCast(count);
            }
            // No table - use a default batch size
            break :blk 1000;
        };

        // Check if this is a batch method
        if (dispatcher.isBatchMethod(class_name, mc.method)) {
            // Allocate output buffer
            var output = logic_table_dispatch.ColumnBuffer.initFloat64(self.allocator, row_count) catch
                return error.OutOfMemory;
            errdefer output.deinit(self.allocator);

            const inputs = &[_]logic_table_dispatch.ColumnBinding{};

            // Call batch dispatch
            dispatcher.callMethodBatch(class_name, mc.method, inputs, null, &output, null) catch |err| {
                return switch (err) {
                    logic_table_dispatch.DispatchError.MethodNotFound => error.MethodNotFound,
                    logic_table_dispatch.DispatchError.ArgumentCountMismatch => error.ArgumentCountMismatch,
                    else => error.ExecutionFailed,
                };
            };

            // Cache the results
            const results = output.f64 orelse return error.NoResults;
            const cache_key_copy = self.allocator.dupe(u8, cache_key) catch
                return error.OutOfMemory;
            self.method_results_cache.put(cache_key_copy, results) catch {
                self.allocator.free(cache_key_copy);
                return error.CachePutFailed;
            };

            // Return the result for this row
            if (row_idx < results.len) {
                return Value{ .float = results[row_idx] };
            }
            return error.RowIndexOutOfBounds;
        }

        // Fallback to scalar dispatch for non-batch methods
        const result = dispatcher.callMethod0(class_name, mc.method) catch |err| {
            return switch (err) {
                logic_table_dispatch.DispatchError.MethodNotFound => error.MethodNotFound,
                logic_table_dispatch.DispatchError.ArgumentCountMismatch => error.ArgumentCountMismatch,
                else => error.ExecutionFailed,
            };
        };

        return Value{ .float = result };
    }

    /// Evaluate an expression column for all filtered indices (delegates to expr_eval)
    fn evaluateExpressionColumn(
        self: *Self,
        item: ast.SelectItem,
        indices: []const u32,
    ) !Result.Column {
        // First, preload any columns referenced in the expression
        var col_names = std.ArrayList([]const u8){};
        defer col_names.deinit(self.allocator);
        try self.extractExprColumnNames(&item.expr, &col_names);
        try self.preloadColumns(col_names.items);

        // Delegate to expr_eval module
        const ctx = self.buildExprContext();
        return expr_eval.evaluateExpressionColumn(ctx, item, indices);
    }

    /// Get all row indices (0, 1, 2, ..., n-1)
    fn getAllIndices(self: *Self) ![]u32 {
        // Get row count (works with both Lance and Parquet)
        const row_count = try self.getRowCount();
        const indices = try self.allocator.alloc(u32, @intCast(row_count));

        for (indices, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }

        return indices;
    }

    // ========================================================================
    // Column Reading
    // ========================================================================

    /// Read columns based on SELECT list and filtered indices
    fn readColumns(
        self: *Self,
        select_list: []const ast.SelectItem,
        indices: []const u32,
    ) ![]Result.Column {
        var columns = std.ArrayList(Result.Column){};
        errdefer {
            for (columns.items) |col| {
                col.data.free(self.allocator);
            }
            columns.deinit(self.allocator);
        }

        for (select_list) |item| {
            if (isWindowFunction(&item.expr)) continue;

            if (item.expr == .column and std.mem.eql(u8, item.expr.column.name, "*")) {
                const col_names_result = try self.getColumnNamesWithOwnership();
                defer col_names_result.deinit(self.allocator);

                for (col_names_result.names) |col_name| {
                    // Look up the physical column ID from the name
                    // The physical column ID maps to the column metadata index
                    const physical_col_id = self.getPhysicalColumnId(col_name) orelse return error.ColumnNotFound;
                    const data = try self.readColumnAtIndices(physical_col_id, indices);

                    try columns.append(self.allocator, Result.Column{
                        .name = col_name,
                        .data = data,
                    });
                }
                break;
            }

            if (item.expr == .column) {
                const col_name = item.expr.column.name;
                const col_idx = self.getPhysicalColumnId(col_name) orelse return error.ColumnNotFound;
                const data = try self.readColumnAtIndices(col_idx, indices);

                try columns.append(self.allocator, Result.Column{
                    .name = item.alias orelse col_name,
                    .data = data,
                });
            } else {
                const expr_col = try self.evaluateExpressionColumn(item, indices);
                try columns.append(self.allocator, expr_col);
            }
        }

        return columns.toOwnedSlice(self.allocator);
    }

    /// Read column data at specific row indices
    fn readColumnAtIndices(
        self: *Self,
        col_idx: u32,
        indices: []const u32,
    ) !Result.ColumnData {
        // Use generic reader for typed tables
        inline for (typed_table_fields) |field| {
            if (@field(self, field)) |t| return self.readColumnFromTableAtIndices(t, col_idx, indices);
        }

        const fld = self.tbl().getFieldById(col_idx) orelse return error.InvalidColumn;
        const col_type = LanceColumnType.fromLogicalType(fld.logical_type);

        // String requires special handling for ownership
        if (col_type == .string) {
            const all_data = try self.tbl().readStringColumn(col_idx);
            defer {
                for (all_data) |str| self.allocator.free(str);
                self.allocator.free(all_data);
            }
            const filtered = try self.allocator.alloc([]const u8, indices.len);
            for (indices, 0..) |idx, i| filtered[i] = try self.allocator.dupe(u8, all_data[idx]);
            return Result.ColumnData{ .string = filtered };
        }

        return switch (col_type) {
            .timestamp_ns => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_ns = try self.filterByIndices(i64, all_data, indices) };
            },
            .timestamp_us => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_us = try self.filterByIndices(i64, all_data, indices) };
            },
            .timestamp_ms => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_ms = try self.filterByIndices(i64, all_data, indices) };
            },
            .timestamp_s => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .timestamp_s = try self.filterByIndices(i64, all_data, indices) };
            },
            .date32 => blk: {
                const all_data = try self.tbl().readInt32Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .date32 = try self.filterByIndices(i32, all_data, indices) };
            },
            .date64 => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .date64 = try self.filterByIndices(i64, all_data, indices) };
            },
            .int32 => blk: {
                const all_data = try self.tbl().readInt32Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .int32 = try self.filterByIndices(i32, all_data, indices) };
            },
            .float32 => blk: {
                const all_data = try self.tbl().readFloat32Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .float32 = try self.filterByIndices(f32, all_data, indices) };
            },
            .bool_ => blk: {
                const all_data = try self.tbl().readBoolColumn(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .bool_ = try self.filterByIndices(bool, all_data, indices) };
            },
            .int64 => blk: {
                const all_data = try self.tbl().readInt64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .int64 = try self.filterByIndices(i64, all_data, indices) };
            },
            .float64 => blk: {
                const all_data = try self.tbl().readFloat64Column(col_idx);
                defer self.allocator.free(all_data);
                break :blk Result.ColumnData{ .float64 = try self.filterByIndices(f64, all_data, indices) };
            },
            .string => unreachable, // Handled above
            .unsupported => error.UnsupportedColumnType,
        };
    }

    /// Generic column reader for any table type with standard interface
    fn readColumnFromTableAtIndices(
        self: *Self,
        table: anytype,
        col_idx: u32,
        indices: []const u32,
    ) !Result.ColumnData {
        const T = @TypeOf(table.*);
        const is_xlsx = T == XlsxTable;
        const col_type = table.getColumnType(col_idx) orelse return error.InvalidColumn;

        switch (col_type) {
            .int64 => {
                const all_data = table.readInt64Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .int64 = try self.filterByIndices(i64, all_data, indices) };
            },
            .int32 => {
                const all_data = table.readInt32Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .int32 = try self.filterByIndices(i32, all_data, indices) };
            },
            .double => {
                const all_data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .float64 = try self.filterByIndices(f64, all_data, indices) };
            },
            .float => {
                const all_data = table.readFloat32Column(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .float32 = try self.filterByIndices(f32, all_data, indices) };
            },
            .boolean => {
                const all_data = table.readBoolColumn(col_idx) catch return error.ColumnReadError;
                defer self.allocator.free(all_data);
                return Result.ColumnData{ .bool_ = try self.filterByIndices(bool, all_data, indices) };
            },
            .byte_array, .fixed_len_byte_array => {
                const all_data = table.readStringColumn(col_idx) catch return error.ColumnReadError;
                defer {
                    for (all_data) |str| self.allocator.free(str);
                    self.allocator.free(all_data);
                }
                const filtered = try self.allocator.alloc([]const u8, indices.len);
                for (indices, 0..) |idx, i| filtered[i] = try self.allocator.dupe(u8, all_data[idx]);
                return Result.ColumnData{ .string = filtered };
            },
            else => {
                if (is_xlsx) {
                    const all_data = table.readFloat64Column(col_idx) catch return error.ColumnReadError;
                    defer self.allocator.free(all_data);
                    return Result.ColumnData{ .float64 = try self.filterByIndices(f64, all_data, indices) };
                }
                return error.UnsupportedColumnType;
            },
        }
    }

    fn freeColumnData(self: *Self, data: *Result.ColumnData) void {
        data.free(self.allocator);
    }

};

/// Get table name from a TableRef (uses alias if available, otherwise the actual name)
fn getTableRefName(ref: *const ast.TableRef) []const u8 {
    return switch (ref.*) {
        .simple => |s| s.alias orelse s.name,
        .function => |f| f.alias orelse "func",
        .join => |j| getTableRefName(j.left), // For nested joins, use the left table name
    };
}

/// Check if an expression is a simple column reference (no computation needed)
fn isSimpleColumnRef(expr: *const ast.Expr) bool {
    return switch (expr.*) {
        .column => true,
        else => false,
    };
}

/// Check if an expression contains arithmetic or string operations
fn isArithmeticExpr(expr: *const ast.Expr) bool {
    return switch (expr.*) {
        .binary => |bin| switch (bin.op) {
            .add, .subtract, .multiply, .divide, .concat => true,
            else => false,
        },
        else => false,
    };
}

/// Check if an expression contains a function call (recursively)
fn hasFunctionCall(expr: *const ast.Expr) bool {
    return switch (expr.*) {
        .call => true,
        .binary => |bin| hasFunctionCall(bin.left) or hasFunctionCall(bin.right),
        .unary => |un| hasFunctionCall(un.operand),
        else => false,
    };
}

/// Check if a function name is an aggregate function
fn isAggregateFn(name: []const u8) bool {
    const agg_funcs = [_][]const u8{
        "COUNT",       "SUM",         "AVG",         "MIN",          "MAX",
        "count",       "sum",         "avg",         "min",          "max",
        "STDDEV",      "STDDEV_POP",  "STDDEV_SAMP", "VARIANCE",     "VAR_POP",
        "VAR_SAMP",    "MEDIAN",      "PERCENTILE",  "PERCENTILE_CONT",
        "stddev",      "stddev_pop",  "stddev_samp", "variance",     "var_pop",
        "var_samp",    "median",      "percentile",  "percentile_cont",
    };
    for (agg_funcs) |func| {
        if (std.mem.eql(u8, name, func)) return true;
    }
    return false;
}

/// Check if an expression contains only aggregate functions (no scalar functions)
fn hasOnlyAggregateFunctions(expr: *const ast.Expr) bool {
    return switch (expr.*) {
        .call => |call| isAggregateFn(call.name),
        .binary => |bin| hasOnlyAggregateFunctions(bin.left) and hasOnlyAggregateFunctions(bin.right),
        .unary => |un| hasOnlyAggregateFunctions(un.operand),
        .column, .value => true,
        else => false,
    };
}

/// Infer the result type of an expression
fn inferExprType(expr: *const ast.Expr, type_map: *const std.StringHashMap(plan_nodes.ColumnType)) plan_nodes.ColumnType {
    return switch (expr.*) {
        .column => |col| type_map.get(col.name) orelse .unknown,
        .value => |val| switch (val) {
            .integer => .i64,
            .float => .f64,
            .string => .string,
            .null, .blob, .parameter => .unknown,
        },
        .binary => |bin| {
            // String concatenation returns string
            if (bin.op == .concat) return .string;
            // Arithmetic operations: promote to f64 if any operand is float
            const left_type = inferExprType(bin.left, type_map);
            const right_type = inferExprType(bin.right, type_map);
            if (left_type == .f64 or right_type == .f64) return .f64;
            if (left_type == .f32 or right_type == .f32) return .f64;
            // Division always returns f64
            if (bin.op == .divide) return .f64;
            return .i64;
        },
        .unary => |un| inferExprType(un.operand, type_map),
        .call => |func| {
            // Infer function return type
            const name = func.name;
            if (std.mem.eql(u8, name, "COUNT")) return .i64;
            if (std.mem.eql(u8, name, "SUM")) return .f64;
            if (std.mem.eql(u8, name, "AVG")) return .f64;
            if (std.mem.eql(u8, name, "MIN") or std.mem.eql(u8, name, "MAX")) {
                if (func.args.len > 0) return inferExprType(&func.args[0], type_map);
                return .f64;
            }
            if (std.mem.eql(u8, name, "LENGTH")) return .i64;
            if (std.mem.eql(u8, name, "ABS")) return .f64;
            if (std.mem.eql(u8, name, "UPPER") or std.mem.eql(u8, name, "LOWER")) return .string;
            if (std.mem.eql(u8, name, "ROUND") or std.mem.eql(u8, name, "CEIL") or std.mem.eql(u8, name, "FLOOR")) return .f64;
            // Date/time functions all return i64
            if (std.mem.eql(u8, name, "YEAR") or std.mem.eql(u8, name, "MONTH") or std.mem.eql(u8, name, "DAY")) return .i64;
            if (std.mem.eql(u8, name, "HOUR") or std.mem.eql(u8, name, "MINUTE") or std.mem.eql(u8, name, "SECOND")) return .i64;
            if (std.mem.eql(u8, name, "DAYOFWEEK") or std.mem.eql(u8, name, "DAYOFYEAR")) return .i64;
            if (std.mem.eql(u8, name, "DATE_TRUNC") or std.mem.eql(u8, name, "DATE_ADD") or std.mem.eql(u8, name, "EPOCH")) return .i64;
            if (std.mem.eql(u8, name, "QUARTER")) return .i64;
            return .f64; // Default to f64 for unknown functions
        },
        else => .f64, // Default to f64 for complex expressions
    };
}

/// Find the maximum parameter index used in a SELECT statement
/// Returns null if no parameters are used
fn findMaxParamIndex(stmt: *const ast.SelectStmt) ?usize {
    var max_idx: ?usize = null;

    // Check WHERE clause
    if (stmt.where) |*where| {
        if (findMaxParamInExpr(where)) |idx| {
            max_idx = if (max_idx) |m| @max(m, idx) else idx;
        }
    }

    // Check SELECT columns
    for (stmt.columns) |col| {
        if (findMaxParamInExpr(&col.expr)) |idx| {
            max_idx = if (max_idx) |m| @max(m, idx) else idx;
        }
    }

    return max_idx;
}

/// Find the maximum parameter index in an expression tree
fn findMaxParamInExpr(expr: *const ast.Expr) ?usize {
    switch (expr.*) {
        .value => |val| {
            if (val == .parameter) return val.parameter;
            return null;
        },
        .binary => |bin| {
            const left = findMaxParamInExpr(bin.left);
            const right = findMaxParamInExpr(bin.right);
            if (left) |l| {
                if (right) |r| return @max(l, r);
                return l;
            }
            return right;
        },
        .unary => |un| return findMaxParamInExpr(un.operand),
        .call => |func| {
            var max_idx: ?usize = null;
            for (func.args) |*arg| {
                if (findMaxParamInExpr(arg)) |idx| {
                    max_idx = if (max_idx) |m| @max(m, idx) else idx;
                }
            }
            return max_idx;
        },
        else => return null,
    }
}

// Tests are in tests/test_sql_executor.zig
