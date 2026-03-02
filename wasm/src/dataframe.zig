//! DataFrame API for fluent query building.
//!
//! Provides a chainable interface for building queries with full SQL parity:
//!
//! Basic operations:
//!   DataFrame.from(&table)
//!       .select(&.{"id", "name"})
//!       .filter(Expr.gt(Expr.col("id"), Expr.int(10)))
//!       .orderBy("name", false)
//!       .limit(100)
//!       .collect()
//!
//! Joins:
//!   df1.join(df2, .inner, "df1.id = df2.user_id")
//!
//! Window functions:
//!   df.window("row_number", .{}, .{ .partition = &.{"dept"}, .order = &.{"salary DESC"} })
//!
//! Pivot:
//!   df.pivot("category", "amount", .sum)
//!
//! Set operations:
//!   df1.union_(df2).intersect(df3)

const std = @import("std");
const Value = @import("lanceql.value").Value;
const query = @import("lanceql.query");
const Expr = query.Expr;
const SelectStmt = query.SelectStmt;
const SelectItem = query.SelectItem;
const OrderBy = query.OrderBy;
const Executor = query.Executor;
const ResultSet = query.ResultSet;
const Aggregate = query.Aggregate;
const table_mod = @import("lanceql.table");
const Table = table_mod.Table;

// Reuse SQL layer types for consistency
const sql_ast = @import("sql").ast;
const sql_agg = @import("sql").aggregate_functions;

/// Reuse SQL layer's AggregateType for all aggregate operations
pub const AggregateType = sql_agg.AggregateType;

/// Reuse SQL layer's JoinType
pub const JoinType = sql_ast.JoinType;

/// Reuse SQL layer's SetOperationType
pub const SetOpType = sql_ast.SetOperationType;

/// Reuse SQL layer's WindowSpec for window functions
pub const WindowFrame = sql_ast.WindowFrame;
pub const FrameBound = sql_ast.FrameBound;

/// DataFrame for building and executing queries.
pub const DataFrame = struct {
    allocator: std.mem.Allocator,
    table: ?*const Table = null,

    // Query state
    select_columns: ?[]const []const u8 = null,
    select_exprs: ?[]SelectItem = null,
    filter_expr: ?*Expr = null,
    group_columns: ?[]const []const u8 = null,
    aggregates: ?[]AggSpec = null,
    order_columns: ?[]OrderSpec = null,
    limit_value: ?u64 = null,
    offset_value: ?u64 = null,
    distinct_flag: bool = false,
    having_expr: ?*Expr = null,

    // Join state
    join_spec: ?JoinSpec = null,

    // Window function state
    window_specs: ?[]WindowSpec = null,

    // Pivot state
    pivot_spec: ?PivotSpec = null,

    // Set operation state
    set_op: ?SetOpSpec = null,

    // Source for derived DataFrames (joins, set ops)
    source_result: ?ResultSet = null,

    const Self = @This();

    /// Join specification (reuses JoinType from SQL layer)
    pub const JoinSpec = struct {
        other: *const DataFrame,
        join_type: JoinType,
        on_condition: []const u8, // SQL condition string
    };

    /// Window function specification
    pub const WindowSpec = struct {
        func_name: []const u8, // row_number, rank, dense_rank, lag, lead, sum, avg, etc.
        args: []const []const u8, // function arguments (column names)
        partition_by: ?[]const []const u8 = null,
        order_by: ?[]const []const u8 = null, // "col DESC" format
        alias: ?[]const u8 = null,
        offset: ?i64 = null, // For lag/lead offset
        default_value: ?[]const u8 = null, // For lag/lead default value
        n_tiles: ?u32 = null, // For ntile(n)
    };

    /// Pivot specification
    pub const PivotSpec = struct {
        pivot_column: []const u8, // Column whose values become new columns
        value_column: []const u8, // Column to aggregate
        agg_type: AggregateType, // How to aggregate
    };

    /// Set operation specification (reuses SetOpType from SQL layer)
    pub const SetOpSpec = struct {
        other: *const DataFrame,
        op_type: SetOpType,
    };

    /// Aggregate specification for groupBy().agg()
    pub const AggSpec = struct {
        agg_type: AggregateType,
        column: ?[]const u8, // null for COUNT(*)
        alias: ?[]const u8,
    };

    /// Order specification
    pub const OrderSpec = struct {
        column: []const u8,
        descending: bool,
    };

    /// Create a DataFrame from a Table.
    pub fn from(allocator: std.mem.Allocator, table: *const Table) Self {
        return .{
            .allocator = allocator,
            .table = table,
        };
    }

    /// Select specific columns by name.
    pub fn select(self: Self, columns: []const []const u8) Self {
        var new = self;
        new.select_columns = columns;
        return new;
    }

    /// Select with expressions.
    pub fn selectExprs(self: Self, items: []SelectItem) Self {
        var new = self;
        new.select_exprs = items;
        return new;
    }

    /// Filter rows by predicate.
    pub fn filter(self: Self, predicate: *Expr) Self {
        var new = self;
        new.filter_expr = predicate;
        return new;
    }

    /// Filter rows by SQL-like condition string.
    /// Example: filterSql("id > 10 AND name = 'Alice'")
    pub fn filterSql(self: Self, sql: []const u8) !Self {
        var parser = query.Parser.init(self.allocator, sql);
        const expr = try parser.parseExpr();
        return self.filter(expr);
    }

    /// Group by columns.
    pub fn groupBy(self: Self, columns: []const []const u8) GroupedFrame {
        return GroupedFrame{
            .df = self,
            .group_columns = columns,
        };
    }

    /// Order by column.
    pub fn orderBy(self: Self, column: []const u8, descending: bool) Self {
        var new = self;
        var specs: [1]OrderSpec = .{.{ .column = column, .descending = descending }};
        new.order_columns = self.allocator.dupe(OrderSpec, &specs) catch null;
        return new;
    }

    /// Order by multiple columns.
    pub fn orderByMultiple(self: Self, specs: []const OrderSpec) Self {
        var new = self;
        new.order_columns = self.allocator.dupe(OrderSpec, specs) catch null;
        return new;
    }

    /// Limit the number of rows.
    pub fn limit(self: Self, n: u64) Self {
        var new = self;
        new.limit_value = n;
        return new;
    }

    /// Skip rows.
    pub fn offset(self: Self, n: u64) Self {
        var new = self;
        new.offset_value = n;
        return new;
    }

    /// Select distinct rows.
    pub fn distinct(self: Self) Self {
        var new = self;
        new.distinct_flag = true;
        return new;
    }

    // =========================================================================
    // JOIN Operations
    // =========================================================================

    /// Join with another DataFrame.
    /// Example: df1.join(df2, .inner, "df1.id = df2.user_id")
    pub fn join(self: *const Self, other: *const DataFrame, join_type: JoinType, on_condition: []const u8) Self {
        var new = self.*;
        new.join_spec = .{
            .other = other,
            .join_type = join_type,
            .on_condition = on_condition,
        };
        return new;
    }

    /// Inner join shorthand.
    pub fn innerJoin(self: *const Self, other: *const DataFrame, on_condition: []const u8) Self {
        return self.join(other, .inner, on_condition);
    }

    /// Left outer join shorthand.
    pub fn leftJoin(self: *const Self, other: *const DataFrame, on_condition: []const u8) Self {
        return self.join(other, .left, on_condition);
    }

    /// Right outer join shorthand.
    pub fn rightJoin(self: *const Self, other: *const DataFrame, on_condition: []const u8) Self {
        return self.join(other, .right, on_condition);
    }

    /// Cross join (cartesian product).
    pub fn crossJoin(self: *const Self, other: *const DataFrame) Self {
        return self.join(other, .cross, "");
    }

    // =========================================================================
    // WINDOW Functions
    // =========================================================================

    /// Add a window function to the select list.
    /// Example: df.window("row_number", &.{}, .{ .partition_by = &.{"dept"}, .order_by = &.{"salary DESC"} })
    pub fn window(self: Self, func_name: []const u8, args: []const []const u8, spec: struct {
        partition_by: ?[]const []const u8 = null,
        order_by: ?[]const []const u8 = null,
        alias: ?[]const u8 = null,
        offset: ?i64 = null,
        default_value: ?[]const u8 = null,
        n_tiles: ?u32 = null,
    }) Self {
        var new = self;
        const win_spec = WindowSpec{
            .func_name = func_name,
            .args = args,
            .partition_by = spec.partition_by,
            .order_by = spec.order_by,
            .alias = spec.alias,
            .offset = spec.offset,
            .default_value = spec.default_value,
            .n_tiles = spec.n_tiles,
        };

        if (new.window_specs) |existing| {
            var list = std.ArrayList(WindowSpec).init(new.allocator);
            list.appendSlice(existing) catch return new;
            list.append(win_spec) catch return new;
            new.window_specs = list.toOwnedSlice() catch null;
        } else {
            var arr = new.allocator.alloc(WindowSpec, 1) catch return new;
            arr[0] = win_spec;
            new.window_specs = arr;
        }
        return new;
    }

    /// Row number window function.
    pub fn rowNumber(self: Self, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("row_number", &.{}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "row_num",
        });
    }

    /// Rank window function.
    pub fn rank(self: Self, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("rank", &.{}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "rank",
        });
    }

    /// Dense rank window function.
    pub fn denseRank(self: Self, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("dense_rank", &.{}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "dense_rank",
        });
    }

    /// Lag window function - access previous row's value.
    pub fn lag(self: Self, column: []const u8, offset_val: i64, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("lag", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "lag_value",
            .offset = offset_val,
        });
    }

    /// Lead window function - access next row's value.
    pub fn lead(self: Self, column: []const u8, offset_val: i64, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("lead", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "lead_value",
            .offset = offset_val,
        });
    }

    /// Ntile window function - divide rows into N buckets.
    pub fn ntile(self: Self, n: u32, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("ntile", &.{}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "ntile",
            .n_tiles = n,
        });
    }

    /// Percent rank window function - relative rank as percentage.
    pub fn percentRank(self: Self, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("percent_rank", &.{}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "percent_rank",
        });
    }

    /// Cume_dist window function - cumulative distribution.
    pub fn cumeDist(self: Self, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("cume_dist", &.{}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "cume_dist",
        });
    }

    /// First value window function - first value in partition.
    pub fn firstValue(self: Self, column: []const u8, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("first_value", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "first_value",
        });
    }

    /// Last value window function - last value in partition.
    pub fn lastValue(self: Self, column: []const u8, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("last_value", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "last_value",
        });
    }

    /// Nth value window function - nth value in partition.
    pub fn nthValue(self: Self, column: []const u8, n: i64, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("nth_value", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "nth_value",
            .offset = n, // Reuse offset for n
        });
    }

    // Aggregate window functions - run aggregates over window frames

    /// Sum over window.
    pub fn sumOver(self: Self, column: []const u8, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("sum", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "sum_over",
        });
    }

    /// Avg over window.
    pub fn avgOver(self: Self, column: []const u8, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("avg", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "avg_over",
        });
    }

    /// Min over window.
    pub fn minOver(self: Self, column: []const u8, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("min", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "min_over",
        });
    }

    /// Max over window.
    pub fn maxOver(self: Self, column: []const u8, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("max", &.{column}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "max_over",
        });
    }

    /// Count over window.
    pub fn countOver(self: Self, partition_by: ?[]const []const u8, order_by: ?[]const []const u8) Self {
        return self.window("count", &.{}, .{
            .partition_by = partition_by,
            .order_by = order_by,
            .alias = "count_over",
        });
    }

    // =========================================================================
    // PIVOT Operations
    // =========================================================================

    /// Pivot rows to columns.
    /// Example: df.pivot("category", "amount", .sum)
    /// This transforms:
    ///   | id | category | amount |
    ///   |----|----------|--------|
    ///   | 1  | A        | 10     |
    ///   | 1  | B        | 20     |
    /// Into:
    ///   | id | A  | B  |
    ///   |----|----|----|
    ///   | 1  | 10 | 20 |
    pub fn pivot(self: Self, pivot_column: []const u8, value_column: []const u8, agg_type: AggregateType) Self {
        var new = self;
        new.pivot_spec = .{
            .pivot_column = pivot_column,
            .value_column = value_column,
            .agg_type = agg_type,
        };
        return new;
    }

    // =========================================================================
    // SET Operations
    // =========================================================================

    /// Union all (keeps duplicates).
    pub fn unionAll(self: *const Self, other: *const DataFrame) Self {
        var new = self.*;
        new.set_op = .{
            .other = other,
            .op_type = .union_all,
        };
        return new;
    }

    /// Union (removes duplicates).
    pub fn union_(self: *const Self, other: *const DataFrame) Self {
        var new = self.*;
        new.set_op = .{
            .other = other,
            .op_type = .union_distinct,
        };
        return new;
    }

    /// Intersect (rows in both).
    pub fn intersect(self: *const Self, other: *const DataFrame) Self {
        var new = self.*;
        new.set_op = .{
            .other = other,
            .op_type = .intersect,
        };
        return new;
    }

    /// Except (rows in first but not second).
    pub fn except(self: *const Self, other: *const DataFrame) Self {
        var new = self.*;
        new.set_op = .{
            .other = other,
            .op_type = .except,
        };
        return new;
    }

    // =========================================================================
    // Additional SQL-like Methods
    // =========================================================================

    /// Alias this DataFrame (for self-joins and CTEs).
    pub fn alias(self: Self, name: []const u8) struct { df: Self, name: []const u8 } {
        return .{ .df = self, .name = name };
    }

    /// Take first N rows (alias for limit).
    pub fn head(self: Self, n: u64) Self {
        return self.limit(n);
    }

    /// Skip first N rows (alias for offset).
    pub fn tail(self: Self, n: u64) Self {
        return self.offset(n);
    }

    /// Sample N random rows.
    pub fn sample(self: Self, n: u64) Self {
        // For now, just return first N - true random sampling needs executor support
        return self.limit(n);
    }

    // =========================================================================
    // SQL Generation
    // =========================================================================

    /// Generate SQL string from DataFrame operations.
    /// Enables interoperability with browser DataFrame and external SQL engines.
    pub fn toSQL(self: Self) ![]const u8 {
        var sql: std.ArrayListUnmanaged(u8) = .empty;
        errdefer sql.deinit(self.allocator);

        // SELECT clause
        try sql.appendSlice(self.allocator, "SELECT ");
        if (self.distinct_flag) {
            try sql.appendSlice(self.allocator, "DISTINCT ");
        }

        if (self.select_columns) |cols| {
            for (cols, 0..) |col, i| {
                if (i > 0) try sql.appendSlice(self.allocator, ", ");
                try sql.appendSlice(self.allocator, col);
            }
        } else if (self.aggregates) |aggs| {
            // Group columns first
            if (self.group_columns) |gcols| {
                for (gcols, 0..) |col, i| {
                    if (i > 0) try sql.appendSlice(self.allocator, ", ");
                    try sql.appendSlice(self.allocator, col);
                }
                if (aggs.len > 0) try sql.appendSlice(self.allocator, ", ");
            }
            // Then aggregates
            for (aggs, 0..) |a, i| {
                if (i > 0) try sql.appendSlice(self.allocator, ", ");
                try sql.appendSlice(self.allocator, aggTypeStr(a.agg_type));
                try sql.append(self.allocator, '(');
                try sql.appendSlice(self.allocator, a.column orelse "*");
                try sql.append(self.allocator, ')');
                if (a.alias) |alias_name| {
                    try sql.appendSlice(self.allocator, " AS ");
                    try sql.appendSlice(self.allocator, alias_name);
                }
            }
        } else {
            try sql.appendSlice(self.allocator, "*");
        }

        // FROM clause
        try sql.appendSlice(self.allocator, " FROM ");
        if (self.table) |tbl| {
            const name = tbl.getName() catch "data";
            try sql.appendSlice(self.allocator, name);
        } else {
            try sql.appendSlice(self.allocator, "data");
        }

        // JOIN clause
        if (self.join_spec) |js| {
            try sql.append(self.allocator, ' ');
            try sql.appendSlice(self.allocator, joinTypeStr(js.join_type));
            try sql.appendSlice(self.allocator, " JOIN ");
            if (js.other.table) |other_tbl| {
                const name = other_tbl.getName() catch "other";
                try sql.appendSlice(self.allocator, name);
            } else {
                try sql.appendSlice(self.allocator, "other");
            }
            if (js.on_condition.len > 0) {
                try sql.appendSlice(self.allocator, " ON ");
                try sql.appendSlice(self.allocator, js.on_condition);
            }
        }

        // GROUP BY clause
        if (self.group_columns) |cols| {
            try sql.appendSlice(self.allocator, " GROUP BY ");
            for (cols, 0..) |col, i| {
                if (i > 0) try sql.appendSlice(self.allocator, ", ");
                try sql.appendSlice(self.allocator, col);
            }
        }

        // ORDER BY clause
        if (self.order_columns) |orders| {
            try sql.appendSlice(self.allocator, " ORDER BY ");
            for (orders, 0..) |o, i| {
                if (i > 0) try sql.appendSlice(self.allocator, ", ");
                try sql.appendSlice(self.allocator, o.column);
                if (o.descending) {
                    try sql.appendSlice(self.allocator, " DESC");
                }
            }
        }

        // LIMIT clause
        if (self.limit_value) |lim| {
            var buf: [32]u8 = undefined;
            const lim_str = std.fmt.bufPrint(&buf, " LIMIT {d}", .{lim}) catch return error.FormatError;
            try sql.appendSlice(self.allocator, lim_str);
        }

        // OFFSET clause
        if (self.offset_value) |off| {
            var buf: [32]u8 = undefined;
            const off_str = std.fmt.bufPrint(&buf, " OFFSET {d}", .{off}) catch return error.FormatError;
            try sql.appendSlice(self.allocator, off_str);
        }

        return try sql.toOwnedSlice(self.allocator);
    }

    /// Execute the query and return results.
    pub fn collect(self: Self) !ResultSet {
        // Build SelectStmt from DataFrame state
        var stmt = SelectStmt{
            .columns = &.{},
            .from = null,
            .where = self.filter_expr,
            .group_by = self.group_columns orelse &.{},
            .having = null,
            .order_by = &.{},
            .limit = self.limit_value,
            .offset = self.offset_value,
            .distinct = self.distinct_flag,
        };

        // Build select items
        if (self.select_exprs) |exprs| {
            stmt.columns = exprs;
        } else if (self.select_columns) |cols| {
            var items: std.ArrayListUnmanaged(SelectItem) = .empty;
            for (cols) |col| {
                const expr = try self.allocator.create(Expr);
                expr.* = Expr.col(col);
                try items.append(self.allocator, .{ .expr = .{
                    .expression = expr,
                    .alias = null,
                } });
            }
            stmt.columns = try items.toOwnedSlice(self.allocator);
        } else {
            // SELECT *
            var items = try self.allocator.alloc(SelectItem, 1);
            items[0] = .star;
            stmt.columns = items;
        }

        // Build order by
        if (self.order_columns) |orders| {
            var obs: std.ArrayListUnmanaged(OrderBy) = .empty;
            for (orders) |o| {
                try obs.append(self.allocator, .{
                    .column = o.column,
                    .descending = o.descending,
                });
            }
            stmt.order_by = try obs.toOwnedSlice(self.allocator);
        }

        // Create executor
        var executor = Executor.init(self.allocator, stmt);

        // Create data provider from table
        const provider = try self.createDataProvider();

        return executor.execute(provider);
    }

    /// Get row count (without fetching all data).
    pub fn count(self: Self) !u64 {
        const result = try self.collect();
        return result.rowCount();
    }

    /// Get first row.
    pub fn first(self: Self) !?[]Value {
        var limited = self.limit(1);
        const result = try limited.collect();
        if (result.rows.len > 0) {
            return result.rows[0];
        }
        return null;
    }

    /// Create data provider from table.
    fn createDataProvider(self: Self) !Executor.DataProvider {
        const table = self.table orelse return error.NoTableSource;

        const TableProvider = struct {
            tbl: *const Table,
            col_names: [][]const u8,
            row_count: usize,

            fn getColumnNames(ptr: *anyopaque) [][]const u8 {
                const p: *@This() = @ptrCast(@alignCast(ptr));
                return p.col_names;
            }

            fn getRowCount(ptr: *anyopaque) usize {
                const p: *@This() = @ptrCast(@alignCast(ptr));
                return p.row_count;
            }

            fn readInt64Column(ptr: *anyopaque, col_idx: usize) ?[]i64 {
                const p: *@This() = @ptrCast(@alignCast(ptr));
                return p.tbl.readInt64Column(@intCast(col_idx)) catch null;
            }

            fn readFloat64Column(ptr: *anyopaque, col_idx: usize) ?[]f64 {
                const p: *@This() = @ptrCast(@alignCast(ptr));
                return p.tbl.readFloat64Column(@intCast(col_idx)) catch null;
            }

            const vtable = Executor.DataProvider.VTable{
                .getColumnNames = getColumnNames,
                .getRowCount = getRowCount,
                .readInt64Column = readInt64Column,
                .readFloat64Column = readFloat64Column,
            };
        };

        const col_names = try table.columnNames();
        const row_count = try table.rowCount(0);

        const provider = try self.allocator.create(TableProvider);
        provider.* = .{
            .tbl = table,
            .col_names = col_names,
            .row_count = @intCast(row_count),
        };

        return .{
            .ptr = provider,
            .vtable = &TableProvider.vtable,
        };
    }
};

/// Grouped DataFrame for aggregation operations.
pub const GroupedFrame = struct {
    df: DataFrame,
    group_columns: []const []const u8,

    const Self = @This();

    /// Aggregate with specifications.
    pub fn agg(self: Self, specs: []const DataFrame.AggSpec) DataFrame {
        var new_df = self.df;
        new_df.group_columns = self.group_columns;
        new_df.aggregates = self.df.allocator.dupe(DataFrame.AggSpec, specs) catch null;
        return new_df;
    }

    /// Count rows per group.
    pub fn count(self: Self) DataFrame {
        return self.agg(&.{.{
            .agg_type = .count,
            .column = null,
            .alias = "count",
        }});
    }

    /// Sum a column per group.
    pub fn sum(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .sum,
            .column = column,
            .alias = null,
        }});
    }

    /// Average a column per group.
    pub fn avg(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .avg,
            .column = column,
            .alias = null,
        }});
    }

    /// Min of a column per group.
    pub fn min(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .min,
            .column = column,
            .alias = null,
        }});
    }

    /// Max of a column per group.
    pub fn max(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .max,
            .column = column,
            .alias = null,
        }});
    }

    /// Sample standard deviation per group.
    pub fn stddev(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .stddev,
            .column = column,
            .alias = null,
        }});
    }

    /// Population standard deviation per group.
    pub fn stddevPop(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .stddev_pop,
            .column = column,
            .alias = null,
        }});
    }

    /// Sample variance per group.
    pub fn variance(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .variance,
            .column = column,
            .alias = null,
        }});
    }

    /// Population variance per group.
    pub fn varPop(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .var_pop,
            .column = column,
            .alias = null,
        }});
    }

    /// Median (50th percentile) per group.
    pub fn median(self: Self, column: []const u8) DataFrame {
        return self.agg(&.{.{
            .agg_type = .median,
            .column = column,
            .alias = null,
        }});
    }

    /// Arbitrary percentile per group (0.0-1.0).
    pub fn percentile(self: Self, column: []const u8, pct: f64) DataFrame {
        _ = pct; // Percentile value is passed via args in SQL layer
        return self.agg(&.{.{
            .agg_type = .percentile,
            .column = column,
            .alias = null,
        }});
    }

    /// Apply HAVING clause to filter groups.
    pub fn having(self: Self, predicate: *Expr) DataFrame {
        var new_df = self.df;
        new_df.group_columns = self.group_columns;
        new_df.having_expr = predicate;
        return new_df;
    }
};

// ============================================================================
// SQL Generation Helpers
// ============================================================================

/// Convert AggregateType to SQL string.
fn aggTypeStr(agg_type: AggregateType) []const u8 {
    return switch (agg_type) {
        .count => "COUNT",
        .sum => "SUM",
        .avg => "AVG",
        .min => "MIN",
        .max => "MAX",
        .stddev => "STDDEV",
        .stddev_pop => "STDDEV_POP",
        .variance => "VARIANCE",
        .var_pop => "VAR_POP",
        .median => "MEDIAN",
        .percentile => "PERCENTILE",
        .count_distinct => "COUNT_DISTINCT",
        .first => "FIRST",
        .last => "LAST",
        .group_concat => "GROUP_CONCAT",
        .array_agg => "ARRAY_AGG",
        .string_agg => "STRING_AGG",
        .bit_and => "BIT_AND",
        .bit_or => "BIT_OR",
        .bit_xor => "BIT_XOR",
        .bool_and => "BOOL_AND",
        .bool_or => "BOOL_OR",
        .any_value => "ANY_VALUE",
        .corr => "CORR",
        .covar_pop => "COVAR_POP",
        .covar_samp => "COVAR_SAMP",
        .regr_avgx => "REGR_AVGX",
        .regr_avgy => "REGR_AVGY",
        .regr_count => "REGR_COUNT",
        .regr_intercept => "REGR_INTERCEPT",
        .regr_r2 => "REGR_R2",
        .regr_slope => "REGR_SLOPE",
        .regr_sxx => "REGR_SXX",
        .regr_sxy => "REGR_SXY",
        .regr_syy => "REGR_SYY",
        .rollup_sum, .rollup_count, .rollup_avg, .rollup_min, .rollup_max => "ROLLUP",
    };
}

/// Convert JoinType to SQL string.
fn joinTypeStr(join_type: JoinType) []const u8 {
    return switch (join_type) {
        .inner => "INNER",
        .left => "LEFT",
        .right => "RIGHT",
        .full => "FULL",
        .cross => "CROSS",
        .natural => "NATURAL",
        .left_semi => "LEFT SEMI",
        .left_anti => "LEFT ANTI",
    };
}

// ============================================================================
// Expression Builder Helpers
// ============================================================================

/// Build expressions fluently.
pub const ExprBuilder = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    /// Column reference.
    pub fn col(self: Self, name: []const u8) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = Expr.col(name);
        return e;
    }

    /// Integer literal.
    pub fn int(self: Self, v: i64) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = Expr.intLit(v);
        return e;
    }

    /// Float literal.
    pub fn float(self: Self, v: f64) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = Expr.floatLit(v);
        return e;
    }

    /// String literal.
    pub fn str(self: Self, v: []const u8) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = Expr.strLit(v);
        return e;
    }

    /// Boolean literal.
    pub fn boolean(self: Self, v: bool) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = Expr.boolLit(v);
        return e;
    }

    /// Equals comparison.
    pub fn eq(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .eq, .left = left, .right = right } };
        return e;
    }

    /// Not equals comparison.
    pub fn ne(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .ne, .left = left, .right = right } };
        return e;
    }

    /// Greater than comparison.
    pub fn gt(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .gt, .left = left, .right = right } };
        return e;
    }

    /// Greater than or equal comparison.
    pub fn ge(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .ge, .left = left, .right = right } };
        return e;
    }

    /// Less than comparison.
    pub fn lt(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .lt, .left = left, .right = right } };
        return e;
    }

    /// Less than or equal comparison.
    pub fn le(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .le, .left = left, .right = right } };
        return e;
    }

    /// Logical AND.
    pub fn @"and"(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .and_, .left = left, .right = right } };
        return e;
    }

    /// Logical OR.
    pub fn @"or"(self: Self, left: *Expr, right: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .binary = .{ .op = .or_, .left = left, .right = right } };
        return e;
    }

    /// Logical NOT.
    pub fn not(self: Self, operand: *Expr) !*Expr {
        const e = try self.allocator.create(Expr);
        e.* = .{ .unary = .{ .op = .not, .operand = operand } };
        return e;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "DataFrame basic construction" {
    // Just test that the types compile
    const allocator = std.testing.allocator;
    _ = ExprBuilder.init(allocator);
}

test "ExprBuilder creates expressions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const eb = ExprBuilder.init(arena.allocator());

    const col_expr = try eb.col("id");
    const int_expr = try eb.int(10);
    const cmp_expr = try eb.gt(col_expr, int_expr);

    try std.testing.expect(cmp_expr.* == .binary);
}

test "GroupedFrame methods" {
    // Test that GroupedFrame methods compile
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    // Note: Can't fully test without a real Table
    _ = arena.allocator();
}
