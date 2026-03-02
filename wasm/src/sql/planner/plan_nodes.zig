//! QueryPlan Intermediate Representation for Fused Query Compilation
//!
//! Represents a query execution plan as a tree of nodes. Each node type
//! corresponds to a relational operator (scan, filter, project, etc.).
//!
//! The plan is built from the SQL AST by the Planner, and then compiled
//! to fused native code by the FusedCodeGen.
//!
//! Example plan for:
//!   SELECT t.risk_score(), amount FROM logic_table('fraud.py') AS t
//!   WHERE amount > 1000 AND t.risk_score() > 0.7 LIMIT 100
//!
//! Limit(100)
//!   └── Project([risk_score, amount])
//!         └── Filter(amount > 1000 AND risk_score > 0.7)
//!               └── Compute(risk_score = FraudDetector.risk_score())
//!                     └── Scan(transactions, [amount, days_since])

const std = @import("std");
const ast = @import("ast");

// ============================================================================
// Column and Type Definitions
// ============================================================================

/// Column data type (resolved from schema)
pub const ColumnType = enum {
    i64,
    i32,
    i16,
    i8,
    u64,
    u32,
    u16,
    u8,
    f64,
    f32,
    bool,
    string,
    bytes,
    vec_f32, // Fixed-size embedding vector
    vec_f64,
    timestamp_ns,
    timestamp_us,
    timestamp_ms,
    timestamp_s,
    date32,
    date64,
    unknown,

    /// Convert to Zig type string for code generation
    pub fn toZigType(self: ColumnType) []const u8 {
        return switch (self) {
            .i64 => "i64",
            .i32 => "i32",
            .i16 => "i16",
            .i8 => "i8",
            .u64 => "u64",
            .u32 => "u32",
            .u16 => "u16",
            .u8 => "u8",
            .f64 => "f64",
            .f32 => "f32",
            .bool => "bool",
            .string => "[]const u8",
            .bytes => "[]const u8",
            .vec_f32 => "[]const f32",
            .vec_f64 => "[]const f64",
            .timestamp_ns, .timestamp_us, .timestamp_ms, .timestamp_s => "i64",
            .date32 => "i32",
            .date64 => "i64",
            .unknown => "f64", // Default to f64 for code generation
        };
    }

    /// Check if this is a numeric type
    pub fn isNumeric(self: ColumnType) bool {
        return switch (self) {
            .i64, .i32, .i16, .i8, .u64, .u32, .u16, .u8, .f64, .f32 => true,
            else => false,
        };
    }

    /// Check if this is a floating point type
    pub fn isFloat(self: ColumnType) bool {
        return self == .f64 or self == .f32;
    }
};

/// Column reference with optional table qualifier and resolved type
pub const ColumnRef = struct {
    /// Table alias or name (null for unqualified columns)
    table: ?[]const u8,
    /// Column name
    column: []const u8,
    /// Resolved column type (from schema)
    col_type: ColumnType = .unknown,
    /// Physical column index in source table
    physical_idx: ?u32 = null,

    /// Format as "table.column" or just "column"
    pub fn format(self: ColumnRef, buf: []u8) ![]const u8 {
        if (self.table) |t| {
            return std.fmt.bufPrint(buf, "{s}.{s}", .{ t, self.column });
        } else {
            return std.fmt.bufPrint(buf, "{s}", .{self.column});
        }
    }

    /// Check equality
    pub fn eql(self: ColumnRef, other: ColumnRef) bool {
        const table_eq = if (self.table) |t1| blk: {
            break :blk if (other.table) |t2| std.mem.eql(u8, t1, t2) else false;
        } else other.table == null;

        return table_eq and std.mem.eql(u8, self.column, other.column);
    }
};

/// Column definition (from schema)
pub const ColumnDef = struct {
    name: []const u8,
    col_type: ColumnType,
    nullable: bool = true,
};

// ============================================================================
// Table Source Definitions
// ============================================================================

/// Type of table source
pub const TableSourceType = enum {
    lance,
    parquet,
    delta,
    iceberg,
    arrow,
    avro,
    orc,
    xlsx,
    csv,
    logic_table, // Virtual table from @logic_table Python
    subquery,
};

/// @logic_table method information
pub const LogicTableMethodInfo = struct {
    /// Method name (e.g., "risk_score")
    name: []const u8,
    /// Python class name (e.g., "FraudDetector")
    class_name: []const u8,
    /// Column dependencies
    column_deps: []const ColumnRef,
    /// Return type
    return_type: ColumnType,
    /// Inlined Zig source (from metal0 compilation)
    inlined_body: ?[]const u8 = null,
};

/// @logic_table information
pub const LogicTableInfo = struct {
    /// Path to Python file
    python_file: []const u8,
    /// Class name (extracted from Python)
    class_name: []const u8,
    /// Available methods
    methods: []const LogicTableMethodInfo,
};

/// Table source information
pub const TableSourceInfo = struct {
    /// Source type
    source_type: TableSourceType,
    /// Table name or alias
    name: []const u8,
    /// Path to data file (for file-based sources)
    path: ?[]const u8,
    /// Schema columns
    schema: []const ColumnDef,
    /// For @logic_table: Python class info
    logic_table_info: ?LogicTableInfo = null,
};

// ============================================================================
// Plan Node Types
// ============================================================================

/// Aggregate function type
pub const AggregateType = enum {
    count,
    count_distinct,
    sum,
    avg,
    min,
    max,
    stddev,
    stddev_pop,
    variance,
    var_pop,
    median,
    percentile,
    first,
    last,
    array_agg,
    string_agg,
};

/// Aggregate specification
pub const AggregateSpec = struct {
    /// Output column name
    name: []const u8,
    /// Aggregate type
    agg_type: AggregateType,
    /// Input column (null for COUNT(*))
    input_col: ?ColumnRef,
    /// DISTINCT flag
    distinct: bool = false,
    /// Percentile value (0.0-1.0) for PERCENTILE function
    percentile_value: f64 = 0.5,
};

/// Window function type
pub const WindowFuncType = enum {
    row_number,
    rank,
    dense_rank,
    ntile,
    lead,
    lag,
    first_value,
    last_value,
    nth_value,
    percent_rank,
    cume_dist,
};

/// Window function specification
pub const WindowFuncSpec = struct {
    /// Output column name
    name: []const u8,
    /// Window function type
    func_type: WindowFuncType,
    /// Partition columns
    partition_by: []const ColumnRef,
    /// Order by columns
    order_by: []const OrderBySpec,
    /// Frame specification (if any)
    frame: ?ast.WindowFrame = null,
};

/// Order by specification
pub const OrderBySpec = struct {
    column: ColumnRef,
    direction: ast.OrderDirection,
    nulls_first: bool = false,
};

/// Join type
pub const JoinType = enum {
    inner,
    left,
    right,
    full,
    cross,
};

/// Computed expression (for @logic_table methods or derived columns)
pub const ComputeExpr = struct {
    /// Output column name
    name: []const u8,
    /// Original AST expression
    expr: *const ast.Expr,
    /// Column dependencies (for codegen)
    deps: []const ColumnRef,
    /// Return type (resolved)
    return_type: ColumnType = .unknown,
    /// Inlined Zig source (from metal0 compilation, for @logic_table methods)
    inlined_body: ?[]const u8 = null,
    /// Is this a @logic_table method call?
    is_logic_table_method: bool = false,
    /// Class name (for @logic_table methods)
    class_name: ?[]const u8 = null,
    /// Method name (for @logic_table methods)
    method_name: ?[]const u8 = null,
};

// ============================================================================
// Plan Nodes
// ============================================================================

/// Query plan node (recursive tree structure)
pub const PlanNode = union(enum) {
    /// Scan table columns from data source
    scan: ScanNode,

    /// Filter rows (WHERE clause)
    filter: FilterNode,

    /// Project columns (SELECT list, final output)
    project: ProjectNode,

    /// Compute expressions (computed columns, @logic_table methods)
    compute: ComputeNode,

    /// Group by with aggregations
    group_by: GroupByNode,

    /// Window function evaluation
    window: WindowNode,

    /// Sort (ORDER BY)
    sort: SortNode,

    /// Limit and offset
    limit: LimitNode,

    /// Hash join (for multi-table queries)
    hash_join: HashJoinNode,

    /// Get the input node (if any)
    pub fn getInput(self: *const PlanNode) ?*const PlanNode {
        return switch (self.*) {
            .scan => null,
            .filter => |f| f.input,
            .project => |p| p.input,
            .compute => |c| c.input,
            .group_by => |g| g.input,
            .window => |w| w.input,
            .sort => |s| s.input,
            .limit => |l| l.input,
            .hash_join => null, // Has two inputs
        };
    }

    /// Get output columns
    pub fn getOutputColumns(self: *const PlanNode) []const ColumnRef {
        return switch (self.*) {
            .scan => |s| s.output_columns,
            .filter => |f| f.input.getOutputColumns(),
            .project => |p| p.output_columns,
            .compute => |c| c.output_columns,
            .group_by => |g| g.output_columns,
            .window => |w| w.input.getOutputColumns(), // TODO: add window columns
            .sort => |s| s.input.getOutputColumns(),
            .limit => |l| l.input.getOutputColumns(),
            .hash_join => |j| j.output_columns,
        };
    }
};

/// Scan node - read columns from data source
pub const ScanNode = struct {
    /// Table source information
    source: TableSourceInfo,

    /// Columns to read from source
    columns: []const ColumnRef,

    /// Output column references (with resolved types)
    output_columns: []const ColumnRef,

    /// Pushed-down predicates (for predicate pushdown optimization)
    pushed_predicates: ?*const ast.Expr = null,

    /// Estimated row count (for cost-based optimization)
    estimated_rows: ?u64 = null,
};

/// Filter node - apply predicate to filter rows
pub const FilterNode = struct {
    /// Input plan node
    input: *const PlanNode,

    /// Filter predicate (AST expression)
    predicate: *const ast.Expr,

    /// Selectivity estimate (0.0 to 1.0)
    selectivity_estimate: f64 = 0.5,

    /// Can this predicate be pushed down to scan?
    pushable: bool = false,
};

/// Project node - select and reorder output columns
pub const ProjectNode = struct {
    /// Input plan node
    input: *const PlanNode,

    /// Output column references
    output_columns: []const ColumnRef,

    /// Expressions for each output column (for computed projections)
    expressions: ?[]const *const ast.Expr = null,

    /// DISTINCT flag - remove duplicate rows
    distinct: bool = false,
};

/// Compute node - evaluate expressions (including @logic_table methods)
pub const ComputeNode = struct {
    /// Input plan node
    input: *const PlanNode,

    /// Expressions to compute
    expressions: []const ComputeExpr,

    /// Output columns (input columns + computed columns)
    output_columns: []const ColumnRef,
};

/// Group by node - grouping and aggregation
pub const GroupByNode = struct {
    /// Input plan node
    input: *const PlanNode,

    /// Group by key columns
    group_keys: []const ColumnRef,

    /// Aggregate specifications
    aggregates: []const AggregateSpec,

    /// HAVING clause predicate (optional)
    having: ?*const ast.Expr = null,

    /// Output columns (group keys + aggregates)
    output_columns: []const ColumnRef,
};

/// Window node - window function evaluation
pub const WindowNode = struct {
    /// Input plan node
    input: *const PlanNode,

    /// Window function specifications
    windows: []const WindowFuncSpec,
};

/// Sort node - order by
pub const SortNode = struct {
    /// Input plan node
    input: *const PlanNode,

    /// Order by specifications
    order_by: []const OrderBySpec,
};

/// Limit node - limit and offset
pub const LimitNode = struct {
    /// Input plan node
    input: *const PlanNode,

    /// Maximum rows to return (null = unlimited)
    limit: ?u32,

    /// Rows to skip (null = 0)
    offset: ?u32 = null,
};

/// Hash join node - join two inputs
pub const HashJoinNode = struct {
    /// Left input
    left: *const PlanNode,

    /// Right input
    right: *const PlanNode,

    /// Join type
    join_type: JoinType,

    /// Left join key column
    left_key: ColumnRef,

    /// Right join key column
    right_key: ColumnRef,

    /// Output columns
    output_columns: []const ColumnRef,
};

// ============================================================================
// Query Plan Container
// ============================================================================

/// Complete query plan with metadata
pub const QueryPlan = struct {
    /// Root node of the plan tree
    root: *const PlanNode,

    /// All table sources referenced
    sources: []const TableSourceInfo,

    /// All @logic_table methods to compile
    logic_table_methods: []const LogicTableMethodInfo,

    /// Estimated total cost (for query optimization)
    estimated_cost: ?f64 = null,

    /// Whether this plan can be compiled to fused code
    compilable: bool = true,

    /// Reason if not compilable
    non_compilable_reason: ?[]const u8 = null,
};

// ============================================================================
// Plan Builder Helper
// ============================================================================

/// Helper for building query plans
pub const PlanBuilder = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Create a scan node
    pub fn scan(self: *Self, source: TableSourceInfo, columns: []const ColumnRef) !*PlanNode {
        const node = try self.allocator.create(PlanNode);
        node.* = .{
            .scan = .{
                .source = source,
                .columns = columns,
                .output_columns = columns,
            },
        };
        return node;
    }

    /// Create a filter node
    pub fn filter(self: *Self, input: *const PlanNode, predicate: *const ast.Expr) !*PlanNode {
        const node = try self.allocator.create(PlanNode);
        node.* = .{
            .filter = .{
                .input = input,
                .predicate = predicate,
            },
        };
        return node;
    }

    /// Create a project node
    pub fn project(self: *Self, input: *const PlanNode, columns: []const ColumnRef) !*PlanNode {
        const node = try self.allocator.create(PlanNode);
        node.* = .{
            .project = .{
                .input = input,
                .output_columns = columns,
            },
        };
        return node;
    }

    /// Create a compute node
    pub fn compute(self: *Self, input: *const PlanNode, expressions: []const ComputeExpr, output_cols: []const ColumnRef) !*PlanNode {
        const node = try self.allocator.create(PlanNode);
        node.* = .{
            .compute = .{
                .input = input,
                .expressions = expressions,
                .output_columns = output_cols,
            },
        };
        return node;
    }

    /// Create a limit node
    pub fn limit(self: *Self, input: *const PlanNode, limit_val: ?u32, offset_val: ?u32) !*PlanNode {
        const node = try self.allocator.create(PlanNode);
        node.* = .{
            .limit = .{
                .input = input,
                .limit = limit_val,
                .offset = offset_val,
            },
        };
        return node;
    }

    /// Create a sort node
    pub fn sort(self: *Self, input: *const PlanNode, order_by: []const OrderBySpec) !*PlanNode {
        const node = try self.allocator.create(PlanNode);
        node.* = .{
            .sort = .{
                .input = input,
                .order_by = order_by,
            },
        };
        return node;
    }

    /// Create a group by node
    pub fn groupBy(
        self: *Self,
        input: *const PlanNode,
        keys: []const ColumnRef,
        aggs: []const AggregateSpec,
        output_cols: []const ColumnRef,
    ) !*PlanNode {
        const node = try self.allocator.create(PlanNode);
        node.* = .{
            .group_by = .{
                .input = input,
                .group_keys = keys,
                .aggregates = aggs,
                .output_columns = output_cols,
            },
        };
        return node;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ColumnType.toZigType" {
    try std.testing.expectEqualStrings("f64", ColumnType.f64.toZigType());
    try std.testing.expectEqualStrings("i64", ColumnType.i64.toZigType());
    try std.testing.expectEqualStrings("bool", ColumnType.bool.toZigType());
    try std.testing.expectEqualStrings("[]const u8", ColumnType.string.toZigType());
}

test "ColumnRef.eql" {
    const ref1 = ColumnRef{ .table = "t", .column = "amount" };
    const ref2 = ColumnRef{ .table = "t", .column = "amount" };
    const ref3 = ColumnRef{ .table = "t", .column = "days" };
    const ref4 = ColumnRef{ .table = null, .column = "amount" };

    try std.testing.expect(ref1.eql(ref2));
    try std.testing.expect(!ref1.eql(ref3));
    try std.testing.expect(!ref1.eql(ref4));
}

test "PlanBuilder basic" {
    const allocator = std.testing.allocator;
    var builder = PlanBuilder.init(allocator);

    // Create a simple scan -> filter -> project plan
    const columns = [_]ColumnRef{
        .{ .table = null, .column = "amount", .col_type = .f64 },
        .{ .table = null, .column = "days", .col_type = .i64 },
    };

    const source = TableSourceInfo{
        .source_type = .lance,
        .name = "transactions",
        .path = "transactions.lance",
        .schema = &.{
            .{ .name = "amount", .col_type = .f64 },
            .{ .name = "days", .col_type = .i64 },
        },
    };

    const scan_node = try builder.scan(source, &columns);
    defer allocator.destroy(scan_node);

    try std.testing.expect(scan_node.* == .scan);
    try std.testing.expectEqual(@as(usize, 2), scan_node.scan.columns.len);
}
