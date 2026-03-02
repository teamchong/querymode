//! Query Planner - Converts SQL AST to QueryPlan
//!
//! The planner transforms a parsed SQL statement (SelectStmt) into an
//! optimized query plan (QueryPlan) that can be compiled to native code.
//!
//! Main responsibilities:
//! 1. Resolve table sources and column references
//! 2. Extract @logic_table method calls for inlining
//! 3. Build plan tree (Scan -> Filter -> Compute -> Project -> ...)
//! 4. Analyze predicate pushdown opportunities
//!
//! Example:
//!   SELECT t.risk_score(), amount
//!   FROM logic_table('fraud.py') AS t
//!   WHERE amount > 1000
//!
//! Produces:
//!   Project([risk_score, amount])
//!     └── Filter(amount > 1000)
//!           └── Compute(risk_score = FraudDetector.risk_score())
//!                 └── Scan(transactions, [amount, days_since])

const std = @import("std");
const plan_nodes = @import("plan_nodes.zig");

// Use module imports for SQL modules
const ast = @import("ast");

const PlanNode = plan_nodes.PlanNode;
const QueryPlan = plan_nodes.QueryPlan;
const ColumnRef = plan_nodes.ColumnRef;
const ColumnType = plan_nodes.ColumnType;
const ColumnDef = plan_nodes.ColumnDef;
const TableSourceInfo = plan_nodes.TableSourceInfo;
const TableSourceType = plan_nodes.TableSourceType;
const ComputeExpr = plan_nodes.ComputeExpr;
const LogicTableInfo = plan_nodes.LogicTableInfo;
const LogicTableMethodInfo = plan_nodes.LogicTableMethodInfo;
const AggregateSpec = plan_nodes.AggregateSpec;
const AggregateType = plan_nodes.AggregateType;
const OrderBySpec = plan_nodes.OrderBySpec;
const PlanBuilder = plan_nodes.PlanBuilder;

/// Planner errors
pub const PlannerError = error{
    OutOfMemory,
    InvalidTableReference,
    UnresolvedColumn,
    UnsupportedExpression,
    UnsupportedTableFunction,
    LogicTableNotFound,
    MethodNotFound,
    AmbiguousColumn,
    InvalidJoin,
};

/// Method call reference (for @logic_table extraction)
const MethodCallRef = struct {
    /// Table alias (e.g., "t")
    object: []const u8,
    /// Method name (e.g., "risk_score")
    method: []const u8,
    /// Original AST expression
    expr: *const ast.Expr,
};

/// Query Planner
pub const Planner = struct {
    allocator: std.mem.Allocator,

    /// Table alias -> source mapping
    table_sources: std.StringHashMap(TableSourceInfo),

    /// @logic_table method references found in query
    method_calls: std.ArrayList(MethodCallRef),

    /// Column binding context (alias.column -> resolved ColumnRef)
    column_bindings: std.StringHashMap(ColumnRef),

    /// Arena for plan node allocations
    arena: std.heap.ArenaAllocator,

    /// Plan sources slice (for cleanup)
    plan_sources: ?[]const TableSourceInfo = null,

    /// Plan methods slice (for cleanup)
    plan_methods: ?[]const LogicTableMethodInfo = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .table_sources = std.StringHashMap(TableSourceInfo).init(allocator),
            .method_calls = .{},
            .column_bindings = std.StringHashMap(ColumnRef).init(allocator),
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        // Free plan slices if allocated
        if (self.plan_sources) |s| self.allocator.free(s);
        if (self.plan_methods) |m| self.allocator.free(m);

        self.table_sources.deinit();
        self.method_calls.deinit(self.allocator);
        self.column_bindings.deinit();
        self.arena.deinit();
    }

    /// Main entry point: convert SelectStmt to QueryPlan
    pub fn plan(self: *Self, stmt: *const ast.SelectStmt) PlannerError!QueryPlan {
        // 1. Resolve table sources from FROM clause
        try self.resolveTableSources(&stmt.from);

        // 2. Extract @logic_table method calls from SELECT and WHERE
        try self.extractMethodCalls(stmt);

        // 3. Build plan tree
        const root = try self.buildPlanTree(stmt);

        // 4. Collect all sources and methods
        var sources: std.ArrayList(TableSourceInfo) = .{};
        var iter = self.table_sources.valueIterator();
        while (iter.next()) |source| {
            sources.append(self.allocator, source.*) catch return PlannerError.OutOfMemory;
        }

        // Build logic table method list
        var methods: std.ArrayList(LogicTableMethodInfo) = .{};
        for (self.method_calls.items) |mc| {
            methods.append(self.allocator, .{
                .name = mc.method,
                .class_name = mc.object, // Will be resolved later
                .column_deps = &.{},
                .return_type = .f64, // Default, will be resolved
            }) catch return PlannerError.OutOfMemory;
        }

        // GROUP BY queries are now compilable (codegen for aggregates implemented)
        _ = stmt.group_by; // Used in plan building

        // Convert to owned slices and free the ArrayList buffers
        const sources_slice = sources.toOwnedSlice(self.allocator) catch return PlannerError.OutOfMemory;
        const methods_slice = methods.toOwnedSlice(self.allocator) catch return PlannerError.OutOfMemory;

        // Store slices for cleanup in deinit
        self.plan_sources = sources_slice;
        self.plan_methods = methods_slice;

        // Check if plan contains hash_join (not yet supported in compiled path)
        const has_join = self.planContainsHashJoin(root);

        return QueryPlan{
            .root = root,
            .sources = sources_slice,
            .logic_table_methods = methods_slice,
            .compilable = !has_join, // JOINs use interpreted path for now
            .non_compilable_reason = if (has_join) "hash_join not yet compiled" else null,
        };
    }

    /// Check if a plan tree contains a hash_join node
    fn planContainsHashJoin(self: *const Self, node: *const plan_nodes.PlanNode) bool {
        return switch (node.*) {
            .hash_join => true,
            .scan => false,
            .filter => |f| self.planContainsHashJoin(f.input),
            .project => |p| self.planContainsHashJoin(p.input),
            .compute => |c| self.planContainsHashJoin(c.input),
            .group_by => |g| self.planContainsHashJoin(g.input),
            .sort => |s| self.planContainsHashJoin(s.input),
            .limit => |l| self.planContainsHashJoin(l.input),
            .window => |w| self.planContainsHashJoin(w.input),
        };
    }

    /// Resolve table sources from FROM clause
    fn resolveTableSources(self: *Self, table_ref: *const ast.TableRef) PlannerError!void {
        switch (table_ref.*) {
            .simple => |simple| {
                const alias = simple.alias orelse simple.name;
                const source = TableSourceInfo{
                    .source_type = self.detectSourceType(simple.name),
                    .name = alias,
                    .path = simple.name,
                    .schema = &.{}, // Will be resolved from file
                };
                self.table_sources.put(alias, source) catch return PlannerError.OutOfMemory;
            },
            .function => |func| {
                // Handle logic_table('path') or other table functions
                if (std.mem.eql(u8, func.func.name, "logic_table")) {
                    if (func.func.args.len == 0) return PlannerError.UnsupportedTableFunction;

                    const path = switch (func.func.args[0]) {
                        .value => |v| switch (v) {
                            .string => |s| s,
                            else => return PlannerError.UnsupportedTableFunction,
                        },
                        else => return PlannerError.UnsupportedTableFunction,
                    };

                    const alias = func.alias orelse "logic_table";
                    const source = TableSourceInfo{
                        .source_type = .logic_table,
                        .name = alias,
                        .path = path,
                        .schema = &.{}, // Will be resolved from Python
                        .logic_table_info = .{
                            .python_file = path,
                            .class_name = "", // Will be extracted
                            .methods = &.{},
                        },
                    };
                    self.table_sources.put(alias, source) catch return PlannerError.OutOfMemory;
                } else {
                    return PlannerError.UnsupportedTableFunction;
                }
            },
            .join => |join| {
                // Resolve both sides of join
                try self.resolveTableSources(join.left);
                try self.resolveTableSources(join.join_clause.table);
            },
        }
    }

    /// Detect source type from file extension
    fn detectSourceType(self: *Self, path: []const u8) TableSourceType {
        _ = self;
        if (std.mem.endsWith(u8, path, ".lance")) return .lance;
        if (std.mem.endsWith(u8, path, ".parquet")) return .parquet;
        if (std.mem.endsWith(u8, path, ".arrow")) return .arrow;
        if (std.mem.endsWith(u8, path, ".avro")) return .avro;
        if (std.mem.endsWith(u8, path, ".orc")) return .orc;
        if (std.mem.endsWith(u8, path, ".csv")) return .csv;
        if (std.mem.endsWith(u8, path, ".xlsx")) return .xlsx;
        return .lance; // Default
    }

    /// Extract @logic_table method calls from statement
    fn extractMethodCalls(self: *Self, stmt: *const ast.SelectStmt) PlannerError!void {
        // Extract from SELECT columns
        for (stmt.columns) |col| {
            try self.extractMethodCallsFromExpr(&col.expr);
        }

        // Extract from WHERE clause
        if (stmt.where) |*where| {
            try self.extractMethodCallsFromExpr(where);
        }

        // Extract from HAVING clause
        if (stmt.group_by) |group_by| {
            if (group_by.having) |*having| {
                try self.extractMethodCallsFromExpr(having);
            }
        }
    }

    /// Extract method calls from expression recursively
    fn extractMethodCallsFromExpr(self: *Self, expr: *const ast.Expr) PlannerError!void {
        switch (expr.*) {
            .method_call => |mc| {
                // Check if object is a known logic_table alias
                if (self.table_sources.get(mc.object)) |source| {
                    if (source.source_type == .logic_table) {
                        self.method_calls.append(self.allocator, .{
                            .object = mc.object,
                            .method = mc.method,
                            .expr = expr,
                        }) catch return PlannerError.OutOfMemory;
                    }
                }

                // Also check method arguments
                for (mc.args) |*arg| {
                    try self.extractMethodCallsFromExpr(arg);
                }
            },
            .binary => |bin| {
                try self.extractMethodCallsFromExpr(bin.left);
                try self.extractMethodCallsFromExpr(bin.right);
            },
            .unary => |un| {
                try self.extractMethodCallsFromExpr(un.operand);
            },
            .call => |call| {
                for (call.args) |*arg| {
                    try self.extractMethodCallsFromExpr(arg);
                }
            },
            .in_list => |in| {
                try self.extractMethodCallsFromExpr(in.expr);
                for (in.values) |*val| {
                    try self.extractMethodCallsFromExpr(val);
                }
            },
            .between => |bet| {
                try self.extractMethodCallsFromExpr(bet.expr);
                try self.extractMethodCallsFromExpr(bet.low);
                try self.extractMethodCallsFromExpr(bet.high);
            },
            .case_expr => |case| {
                if (case.operand) |operand| {
                    try self.extractMethodCallsFromExpr(operand);
                }
                for (case.when_clauses) |*clause| {
                    try self.extractMethodCallsFromExpr(&clause.condition);
                    try self.extractMethodCallsFromExpr(&clause.result);
                }
                if (case.else_result) |else_result| {
                    try self.extractMethodCallsFromExpr(else_result);
                }
            },
            else => {},
        }
    }

    /// Build the plan tree from statement
    fn buildPlanTree(self: *Self, stmt: *const ast.SelectStmt) PlannerError!*const PlanNode {
        const aa = self.arena.allocator();

        // 1. Start with Scan node
        var current = try self.buildScanNode(&stmt.from);

        // 2. Add Compute node for @logic_table methods (if any)
        if (self.method_calls.items.len > 0) {
            current = try self.buildComputeNode(current, stmt);
        }

        // 3. Add Filter node (WHERE clause)
        if (stmt.where) |*where| {
            current = try self.buildFilterNode(current, where);
        }

        // 4. Add GroupBy node (if needed)
        if (stmt.group_by != null or self.hasAggregates(stmt.columns)) {
            current = try self.buildGroupByNode(current, stmt);
        }

        // 5. Add Project node (SELECT columns)
        current = try self.buildProjectNode(current, stmt.columns, aa);

        // 6. Add Sort node (ORDER BY)
        if (stmt.order_by) |order_by| {
            current = try self.buildSortNode(current, order_by);
        }

        // 7. Add Limit node (LIMIT/OFFSET)
        if (stmt.limit != null or stmt.offset != null) {
            current = try self.buildLimitNode(current, stmt.limit, stmt.offset);
        }

        return current;
    }

    /// Build Scan node from table reference
    fn buildScanNode(self: *Self, table_ref: *const ast.TableRef) PlannerError!*const PlanNode {
        const aa = self.arena.allocator();

        switch (table_ref.*) {
            .simple => |simple| {
                const alias = simple.alias orelse simple.name;
                const source = self.table_sources.get(alias) orelse return PlannerError.InvalidTableReference;

                const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
                node.* = .{
                    .scan = .{
                        .source = source,
                        .columns = &.{}, // Will be populated during column resolution
                        .output_columns = &.{},
                    },
                };
                return node;
            },
            .function => |func| {
                const alias = func.alias orelse "logic_table";
                const source = self.table_sources.get(alias) orelse return PlannerError.InvalidTableReference;

                const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
                node.* = .{
                    .scan = .{
                        .source = source,
                        .columns = &.{},
                        .output_columns = &.{},
                    },
                };
                return node;
            },
            .join => |join| {
                // For joins, build both sides and create HashJoinNode
                const left = try self.buildScanNode(join.left);
                const right = try self.buildScanNode(join.join_clause.table);

                // Extract join keys from ON condition
                var left_key = ColumnRef{ .table = null, .column = "" };
                var right_key = ColumnRef{ .table = null, .column = "" };

                if (join.join_clause.on_condition) |on| {
                    // Simple case: col1 = col2
                    if (on == .binary and on.binary.op == .eq) {
                        if (on.binary.left.* == .column) {
                            left_key = .{
                                .table = on.binary.left.column.table,
                                .column = on.binary.left.column.name,
                            };
                        }
                        if (on.binary.right.* == .column) {
                            right_key = .{
                                .table = on.binary.right.column.table,
                                .column = on.binary.right.column.name,
                            };
                        }
                    }
                }

                const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
                node.* = .{
                    .hash_join = .{
                        .left = left,
                        .right = right,
                        .join_type = switch (join.join_clause.join_type) {
                            .inner => .inner,
                            .left => .left,
                            .right => .right,
                            .full => .full,
                            .cross => .cross,
                            .natural => .inner,
                        },
                        .left_key = left_key,
                        .right_key = right_key,
                        .output_columns = &.{},
                    },
                };
                return node;
            },
        }
    }

    /// Build Compute node for @logic_table methods
    fn buildComputeNode(self: *Self, input: *const PlanNode, stmt: *const ast.SelectStmt) PlannerError!*const PlanNode {
        _ = stmt;
        const aa = self.arena.allocator();

        var compute_exprs: std.ArrayList(ComputeExpr) = .{};

        for (self.method_calls.items) |mc| {
            compute_exprs.append(aa, .{
                .name = mc.method,
                .expr = mc.expr,
                .deps = &.{}, // Will be resolved
                .return_type = .f64, // Default
                .is_logic_table_method = true,
                .class_name = mc.object,
                .method_name = mc.method,
            }) catch return PlannerError.OutOfMemory;
        }

        const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
        node.* = .{
            .compute = .{
                .input = input,
                .expressions = compute_exprs.toOwnedSlice(aa) catch return PlannerError.OutOfMemory,
                .output_columns = input.getOutputColumns(),
            },
        };
        return node;
    }

    /// Build Filter node from WHERE clause
    fn buildFilterNode(self: *Self, input: *const PlanNode, predicate: *const ast.Expr) PlannerError!*const PlanNode {
        const aa = self.arena.allocator();

        const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
        node.* = .{
            .filter = .{
                .input = input,
                .predicate = predicate,
                .pushable = self.isPredicatePushable(predicate),
            },
        };
        return node;
    }

    /// Check if predicate can be pushed down to scan
    fn isPredicatePushable(self: *Self, predicate: *const ast.Expr) bool {
        _ = self;
        return switch (predicate.*) {
            .binary => |bin| {
                // Simple comparisons with literals are pushable
                if (bin.op == .eq or bin.op == .ne or bin.op == .lt or bin.op == .le or bin.op == .gt or bin.op == .ge) {
                    const left_is_col = bin.left.* == .column;
                    const right_is_val = bin.right.* == .value;
                    return left_is_col and right_is_val;
                }
                return false;
            },
            else => false,
        };
    }

    /// Build GroupBy node
    fn buildGroupByNode(self: *Self, input: *const PlanNode, stmt: *const ast.SelectStmt) PlannerError!*const PlanNode {
        const aa = self.arena.allocator();

        var group_keys: std.ArrayList(ColumnRef) = .{};
        var aggregates: std.ArrayList(AggregateSpec) = .{};

        // Extract group by columns
        if (stmt.group_by) |group_by| {
            for (group_by.columns) |col| {
                group_keys.append(aa, .{
                    .table = null,
                    .column = col,
                }) catch return PlannerError.OutOfMemory;
            }
        }

        // Extract aggregates from SELECT columns
        for (stmt.columns) |col| {
            if (self.isAggregate(&col.expr)) {
                const agg = try self.extractAggregate(&col.expr, col.alias);
                aggregates.append(aa, agg) catch return PlannerError.OutOfMemory;
            }
        }

        const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
        node.* = .{
            .group_by = .{
                .input = input,
                .group_keys = group_keys.toOwnedSlice(aa) catch return PlannerError.OutOfMemory,
                .aggregates = aggregates.toOwnedSlice(aa) catch return PlannerError.OutOfMemory,
                .having = if (stmt.group_by) |gb| if (gb.having) |*h| h else null else null,
                .output_columns = &.{},
            },
        };
        return node;
    }

    /// Check if expression is an aggregate
    fn isAggregate(self: *Self, expr: *const ast.Expr) bool {
        _ = self;
        return switch (expr.*) {
            .call => |call| {
                const name = call.name;
                return std.mem.eql(u8, name, "COUNT") or
                    std.mem.eql(u8, name, "SUM") or
                    std.mem.eql(u8, name, "AVG") or
                    std.mem.eql(u8, name, "MIN") or
                    std.mem.eql(u8, name, "MAX") or
                    std.mem.eql(u8, name, "STDDEV") or
                    std.mem.eql(u8, name, "STDDEV_SAMP") or
                    std.mem.eql(u8, name, "STDDEV_POP") or
                    std.mem.eql(u8, name, "VARIANCE") or
                    std.mem.eql(u8, name, "VAR_SAMP") or
                    std.mem.eql(u8, name, "VAR_POP") or
                    std.mem.eql(u8, name, "MEDIAN") or
                    std.mem.eql(u8, name, "PERCENTILE");
            },
            else => false,
        };
    }

    /// Check if any column has aggregates
    fn hasAggregates(self: *Self, columns: []const ast.SelectItem) bool {
        for (columns) |col| {
            if (self.isAggregate(&col.expr)) return true;
        }
        return false;
    }

    /// Extract aggregate specification from expression
    fn extractAggregate(self: *Self, expr: *const ast.Expr, alias: ?[]const u8) PlannerError!AggregateSpec {
        _ = self;
        switch (expr.*) {
            .call => |call| {
                const agg_type: AggregateType = if (std.mem.eql(u8, call.name, "COUNT"))
                    if (call.distinct) .count_distinct else .count
                else if (std.mem.eql(u8, call.name, "SUM"))
                    .sum
                else if (std.mem.eql(u8, call.name, "AVG"))
                    .avg
                else if (std.mem.eql(u8, call.name, "MIN"))
                    .min
                else if (std.mem.eql(u8, call.name, "MAX"))
                    .max
                else if (std.mem.eql(u8, call.name, "STDDEV") or std.mem.eql(u8, call.name, "STDDEV_SAMP"))
                    .stddev
                else if (std.mem.eql(u8, call.name, "STDDEV_POP"))
                    .stddev_pop
                else if (std.mem.eql(u8, call.name, "VARIANCE") or std.mem.eql(u8, call.name, "VAR_SAMP"))
                    .variance
                else if (std.mem.eql(u8, call.name, "VAR_POP"))
                    .var_pop
                else if (std.mem.eql(u8, call.name, "MEDIAN"))
                    .median
                else if (std.mem.eql(u8, call.name, "PERCENTILE"))
                    .percentile
                else
                    .count;

                var input_col: ?ColumnRef = null;
                if (call.args.len > 0) {
                    if (call.args[0] == .column) {
                        input_col = .{
                            .table = call.args[0].column.table,
                            .column = call.args[0].column.name,
                        };
                    }
                }

                // Parse percentile value from second argument for PERCENTILE function
                var percentile_value: f64 = 0.5;
                if (agg_type == .percentile and call.args.len > 1) {
                    if (call.args[1] == .value) {
                        const val = call.args[1].value;
                        if (val == .float) {
                            percentile_value = val.float;
                        } else if (val == .integer) {
                            percentile_value = @floatFromInt(val.integer);
                        }
                    }
                }

                return .{
                    .name = alias orelse call.name,
                    .agg_type = agg_type,
                    .input_col = input_col,
                    .distinct = call.distinct,
                    .percentile_value = percentile_value,
                };
            },
            else => return PlannerError.UnsupportedExpression,
        }
    }

    /// Build Project node from SELECT columns
    fn buildProjectNode(self: *Self, input: *const PlanNode, columns: []const ast.SelectItem, aa: std.mem.Allocator) PlannerError!*const PlanNode {
        _ = self;

        var output_cols: std.ArrayList(ColumnRef) = .{};

        for (columns) |col| {
            const col_ref = switch (col.expr) {
                .column => |c| ColumnRef{
                    .table = c.table,
                    .column = c.name,
                },
                .method_call => |mc| ColumnRef{
                    .table = mc.object,
                    .column = col.alias orelse mc.method,
                },
                .call => |call| ColumnRef{
                    .table = null,
                    .column = col.alias orelse call.name,
                },
                else => ColumnRef{
                    .table = null,
                    .column = col.alias orelse "expr",
                },
            };
            output_cols.append(aa, col_ref) catch return PlannerError.OutOfMemory;
        }

        const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
        node.* = .{
            .project = .{
                .input = input,
                .output_columns = output_cols.toOwnedSlice(aa) catch return PlannerError.OutOfMemory,
            },
        };
        return node;
    }

    /// Build Sort node from ORDER BY
    fn buildSortNode(self: *Self, input: *const PlanNode, order_by: []const ast.OrderBy) PlannerError!*const PlanNode {
        const aa = self.arena.allocator();

        var specs: std.ArrayList(OrderBySpec) = .{};
        for (order_by) |ob| {
            specs.append(aa, .{
                .column = .{ .table = null, .column = ob.column },
                .direction = ob.direction,
            }) catch return PlannerError.OutOfMemory;
        }

        const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
        node.* = .{
            .sort = .{
                .input = input,
                .order_by = specs.toOwnedSlice(aa) catch return PlannerError.OutOfMemory,
            },
        };
        return node;
    }

    /// Build Limit node
    fn buildLimitNode(self: *Self, input: *const PlanNode, limit_val: ?u32, offset_val: ?u32) PlannerError!*const PlanNode {
        const aa = self.arena.allocator();

        const node = aa.create(PlanNode) catch return PlannerError.OutOfMemory;
        node.* = .{
            .limit = .{
                .input = input,
                .limit = limit_val,
                .offset = offset_val,
            },
        };
        return node;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Planner init/deinit" {
    const allocator = std.testing.allocator;
    var planner = Planner.init(allocator);
    defer planner.deinit();
}

test "detectSourceType" {
    const allocator = std.testing.allocator;
    var planner = Planner.init(allocator);
    defer planner.deinit();

    try std.testing.expectEqual(TableSourceType.lance, planner.detectSourceType("data.lance"));
    try std.testing.expectEqual(TableSourceType.parquet, planner.detectSourceType("data.parquet"));
    try std.testing.expectEqual(TableSourceType.csv, planner.detectSourceType("data.csv"));
}
