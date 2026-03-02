//! Expression Evaluation - Evaluate SQL expressions to concrete values
//!
//! This module handles arithmetic operations, scalar function calls, and expression evaluation.
//! Uses composition pattern with explicit context parameters.
//!
//! Key design: Uses function pointer for method calls to avoid coupling to executor state.

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;
const Value = ast.Value;
const result_types = @import("result_types.zig");
const Result = result_types.Result;
const CachedColumn = result_types.CachedColumn;
pub const scalar_functions = @import("scalar_functions.zig");
pub const aggregate_functions = @import("aggregate_functions.zig");

/// Result type enum for expression type inference
pub const ResultType = scalar_functions.ResultType;

/// Method call evaluation callback signature
pub const MethodCallFn = *const fn (ctx: *anyopaque, expr: *const Expr, row_idx: u32) anyerror!Value;

/// Expression evaluation context
pub const ExprContext = struct {
    allocator: std.mem.Allocator,
    column_cache: *std.StringHashMap(CachedColumn),
    /// Optional method call handler (for @logic_table methods)
    method_ctx: ?*anyopaque = null,
    method_call_fn: ?MethodCallFn = null,
};

/// Evaluate any expression to a concrete Value for a given row
/// This handles arithmetic, function calls, and nested expressions
pub fn evaluateExprToValue(ctx: ExprContext, expr: *const Expr, row_idx: u32) anyerror!Value {
    return switch (expr.*) {
        .value => expr.value,
        .column => |col| getColumnValue(ctx.column_cache, col.name, row_idx),
        .binary => |bin| evaluateBinaryToValue(ctx, bin, row_idx),
        .unary => |un| evaluateUnaryToValue(ctx, un, row_idx),
        .call => |call| evaluateScalarFunction(ctx, call, row_idx),
        .method_call => blk: {
            if (ctx.method_call_fn) |fn_ptr| {
                break :blk try fn_ptr(ctx.method_ctx.?, expr, row_idx);
            }
            return error.MethodCallsNotSupported;
        },
        else => error.UnsupportedExpression,
    };
}

/// Get value from cached column at row index
pub fn getColumnValue(cache: *std.StringHashMap(CachedColumn), col_name: []const u8, row_idx: u32) !Value {
    const cached = cache.get(col_name) orelse return error.ColumnNotCached;
    return switch (cached) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| Value{ .integer = data[row_idx] },
        .int32, .date32 => |data| Value{ .integer = data[row_idx] },
        .float64 => |data| Value{ .float = data[row_idx] },
        .float32 => |data| Value{ .float = data[row_idx] },
        .bool_ => |data| Value{ .integer = if (data[row_idx]) 1 else 0 },
        .string => |data| Value{ .string = data[row_idx] },
    };
}

/// Evaluate binary expression to a Value (arithmetic operations)
pub fn evaluateBinaryToValue(ctx: ExprContext, bin: anytype, row_idx: u32) anyerror!Value {
    const left = try evaluateExprToValue(ctx, bin.left, row_idx);
    const right = try evaluateExprToValue(ctx, bin.right, row_idx);

    return switch (bin.op) {
        .add => scalar_functions.addValues(left, right),
        .subtract => scalar_functions.subtractValues(left, right),
        .multiply => scalar_functions.multiplyValues(left, right),
        .divide => scalar_functions.divideValues(left, right),
        .concat => concatStrings(ctx.allocator, left, right),
        else => error.UnsupportedOperator,
    };
}

/// Evaluate unary expression to a Value
pub fn evaluateUnaryToValue(ctx: ExprContext, un: anytype, row_idx: u32) anyerror!Value {
    const operand = try evaluateExprToValue(ctx, un.operand, row_idx);

    return switch (un.op) {
        .minus => scalar_functions.negateValue(operand),
        .not => blk: {
            // Boolean negation
            const bool_val = switch (operand) {
                .integer => |i| i != 0,
                .float => |f| f != 0.0,
                .null => false,
                else => true,
            };
            break :blk Value{ .integer = if (bool_val) 0 else 1 };
        },
        else => error.UnsupportedOperator,
    };
}

/// Concatenate two strings (|| operator)
pub fn concatStrings(allocator: std.mem.Allocator, left: Value, right: Value) !Value {
    const left_str = switch (left) {
        .string => |s| s,
        .integer => |i| blk: {
            var buf: [32]u8 = undefined;
            const str = std.fmt.bufPrint(&buf, "{d}", .{i}) catch return error.FormatError;
            break :blk str;
        },
        .float => |f| blk: {
            var buf: [32]u8 = undefined;
            const str = std.fmt.bufPrint(&buf, "{d}", .{f}) catch return error.FormatError;
            break :blk str;
        },
        else => return Value{ .null = {} },
    };

    const right_str = switch (right) {
        .string => |s| s,
        .integer => |i| blk: {
            var buf: [32]u8 = undefined;
            const str = std.fmt.bufPrint(&buf, "{d}", .{i}) catch return error.FormatError;
            break :blk str;
        },
        .float => |f| blk: {
            var buf: [32]u8 = undefined;
            const str = std.fmt.bufPrint(&buf, "{d}", .{f}) catch return error.FormatError;
            break :blk str;
        },
        else => return Value{ .null = {} },
    };

    // Allocate new concatenated string
    const result = try allocator.alloc(u8, left_str.len + right_str.len);
    @memcpy(result[0..left_str.len], left_str);
    @memcpy(result[left_str.len..], right_str);

    return Value{ .string = result };
}

/// Evaluate scalar function call
pub fn evaluateScalarFunction(ctx: ExprContext, call: anytype, row_idx: u32) anyerror!Value {
    // Skip aggregates - handled elsewhere
    if (aggregate_functions.isAggregateFunction(call.name)) {
        return error.AggregateInScalarContext;
    }

    // Evaluate all arguments
    var args: [8]Value = undefined;
    const arg_count = @min(call.args.len, 8);
    for (call.args[0..arg_count], 0..) |*arg, i| {
        args[i] = try evaluateExprToValue(ctx, arg, row_idx);
    }

    // Dispatch by function name (case-insensitive)
    var upper_buf: [32]u8 = undefined;
    const upper_name = std.ascii.upperString(&upper_buf, call.name);

    // String functions
    if (std.mem.eql(u8, upper_name, "UPPER")) {
        return scalar_functions.funcUpper(ctx.allocator, args[0]);
    }
    if (std.mem.eql(u8, upper_name, "LOWER")) {
        return scalar_functions.funcLower(ctx.allocator, args[0]);
    }
    if (std.mem.eql(u8, upper_name, "LENGTH")) {
        return scalar_functions.funcLength(args[0]);
    }
    if (std.mem.eql(u8, upper_name, "TRIM")) {
        return scalar_functions.funcTrim(ctx.allocator, args[0]);
    }

    // Math functions
    if (std.mem.eql(u8, upper_name, "ABS")) {
        return scalar_functions.funcAbs(args[0]);
    }
    if (std.mem.eql(u8, upper_name, "ROUND")) {
        const precision: i32 = if (arg_count > 1) switch (args[1]) {
            .integer => |i| @intCast(i),
            else => 0,
        } else 0;
        return scalar_functions.funcRound(args[0], precision);
    }
    if (std.mem.eql(u8, upper_name, "FLOOR")) {
        return scalar_functions.funcFloor(args[0]);
    }
    if (std.mem.eql(u8, upper_name, "CEIL") or std.mem.eql(u8, upper_name, "CEILING")) {
        return scalar_functions.funcCeil(args[0]);
    }

    // Type functions
    if (std.mem.eql(u8, upper_name, "COALESCE")) {
        // Return first non-null value
        for (args[0..arg_count]) |arg| {
            if (arg != .null) return arg;
        }
        return Value{ .null = {} };
    }

    // Date/Time functions
    if (std.mem.eql(u8, upper_name, "EXTRACT") or std.mem.eql(u8, upper_name, "DATE_PART")) {
        if (arg_count < 2) return Value{ .null = {} };
        return scalar_functions.funcExtract(args[0], args[1]);
    }

    // Shorthand date part extractors: YEAR(ts), MONTH(ts), DAY(ts), etc.
    if (scalar_functions.DatePart.fromFunctionName(upper_name)) |part| {
        return scalar_functions.funcExtractPart(part, args[0]);
    }

    // DATE_TRUNC(part, timestamp)
    if (std.mem.eql(u8, upper_name, "DATE_TRUNC")) {
        if (arg_count < 2) return Value{ .null = {} };
        return scalar_functions.funcDateTrunc(args[0], args[1]);
    }

    // DATE_ADD(timestamp, interval, part) - Add interval to timestamp
    if (std.mem.eql(u8, upper_name, "DATE_ADD") or std.mem.eql(u8, upper_name, "DATEADD")) {
        if (arg_count < 3) return Value{ .null = {} };
        return scalar_functions.funcDateAdd(args[0], args[1], args[2]);
    }

    // DATE_DIFF(timestamp1, timestamp2, part) - Difference between timestamps
    if (std.mem.eql(u8, upper_name, "DATE_DIFF") or std.mem.eql(u8, upper_name, "DATEDIFF")) {
        if (arg_count < 3) return Value{ .null = {} };
        return scalar_functions.funcDateDiff(args[0], args[1], args[2]);
    }

    // EPOCH(timestamp) - Convert to Unix epoch seconds
    if (std.mem.eql(u8, upper_name, "EPOCH") or std.mem.eql(u8, upper_name, "UNIX_TIMESTAMP")) {
        return scalar_functions.funcEpoch(args[0]);
    }

    // FROM_UNIXTIME(epoch_seconds) - Convert Unix epoch to timestamp
    if (std.mem.eql(u8, upper_name, "FROM_UNIXTIME") or std.mem.eql(u8, upper_name, "TO_TIMESTAMP")) {
        return args[0]; // Already an integer, just return as-is
    }

    return error.UnknownFunction;
}

/// Infer the result type of an expression (from cached column types)
pub fn inferExpressionType(ctx: ExprContext, expr: *const Expr) !ResultType {
    return switch (expr.*) {
        .value => |v| switch (v) {
            .integer => .int64,
            .float => .float64,
            .string => .string,
            else => .string,
        },
        .column => |col| blk: {
            const cached = ctx.column_cache.get(col.name) orelse return error.ColumnNotCached;
            // Promote int32/float32 to int64/float64 for expressions
            break :blk switch (cached) {
                .int64, .int32, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date32, .date64, .bool_ => .int64,
                .float64, .float32 => .float64,
                .string => .string,
            };
        },
        .binary => |bin| inferBinaryType(ctx, bin),
        .unary => |un| inferExpressionType(ctx, un.operand),
        .call => |call| scalar_functions.inferFunctionReturnType(call.name),
        else => .string,
    };
}

/// Infer type of binary expression
pub fn inferBinaryType(ctx: ExprContext, bin: anytype) anyerror!ResultType {
    // Concat always returns string
    if (bin.op == .concat) return .string;

    const left_type = try inferExpressionType(ctx, bin.left);
    const right_type = try inferExpressionType(ctx, bin.right);

    // Division always returns float
    if (bin.op == .divide) return .float64;

    // If either operand is float, result is float
    if (left_type == .float64 or right_type == .float64) return .float64;

    // Both integers -> integer
    if (left_type == .int64 and right_type == .int64) return .int64;

    // Default to float for mixed/unknown types
    return .float64;
}

/// Evaluate an expression column for all filtered indices
pub fn evaluateExpressionColumn(
    ctx: ExprContext,
    item: ast.SelectItem,
    indices: []const u32,
) !Result.Column {
    // Infer result type
    const result_type = try inferExpressionType(ctx, &item.expr);

    // Generate column name from expression if no alias
    const col_name = item.alias orelse "expr";

    // Evaluate expression for each row and store results
    switch (result_type) {
        .int64 => {
            const results = try ctx.allocator.alloc(i64, indices.len);
            errdefer ctx.allocator.free(results);

            for (indices, 0..) |row_idx, i| {
                const val = try evaluateExprToValue(ctx, &item.expr, row_idx);
                results[i] = switch (val) {
                    .integer => |v| v,
                    .float => |f| @intFromFloat(f),
                    else => 0,
                };
            }

            return Result.Column{
                .name = col_name,
                .data = Result.ColumnData{ .int64 = results },
            };
        },
        .float64 => {
            const results = try ctx.allocator.alloc(f64, indices.len);
            errdefer ctx.allocator.free(results);

            for (indices, 0..) |row_idx, i| {
                const val = try evaluateExprToValue(ctx, &item.expr, row_idx);
                results[i] = switch (val) {
                    .integer => |v| @floatFromInt(v),
                    .float => |f| f,
                    else => 0.0,
                };
            }

            return Result.Column{
                .name = col_name,
                .data = Result.ColumnData{ .float64 = results },
            };
        },
        .string => {
            const results = try ctx.allocator.alloc([]const u8, indices.len);
            errdefer ctx.allocator.free(results);

            // Check if expression produces owned strings (e.g., concat, UPPER)
            const expr_produces_owned = switch (item.expr) {
                .binary => |bin| bin.op == .concat,
                .call => true, // String functions allocate their results
                else => false,
            };

            for (indices, 0..) |row_idx, i| {
                const val = try evaluateExprToValue(ctx, &item.expr, row_idx);
                results[i] = switch (val) {
                    .string => |s| blk: {
                        if (expr_produces_owned) {
                            break :blk s;
                        } else {
                            break :blk try ctx.allocator.dupe(u8, s);
                        }
                    },
                    .integer => |v| blk: {
                        var buf: [32]u8 = undefined;
                        const str = std.fmt.bufPrint(&buf, "{d}", .{v}) catch "";
                        break :blk try ctx.allocator.dupe(u8, str);
                    },
                    .float => |f| blk: {
                        var buf: [32]u8 = undefined;
                        const str = std.fmt.bufPrint(&buf, "{d}", .{f}) catch "";
                        break :blk try ctx.allocator.dupe(u8, str);
                    },
                    else => try ctx.allocator.dupe(u8, ""),
                };
            }

            return Result.Column{
                .name = col_name,
                .data = Result.ColumnData{ .string = results },
            };
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

test "expr: getColumnValue int64" {
    const allocator = std.testing.allocator;

    var cache = std.StringHashMap(CachedColumn).init(allocator);
    defer cache.deinit();

    const data = try allocator.alloc(i64, 3);
    defer allocator.free(data);
    data[0] = 100;
    data[1] = 200;
    data[2] = 300;

    try cache.put("id", CachedColumn{ .int64 = data });

    const val = try getColumnValue(&cache, "id", 1);
    try std.testing.expectEqual(@as(i64, 200), val.integer);
}

test "expr: evaluateBinaryToValue addition" {
    const allocator = std.testing.allocator;

    var cache = std.StringHashMap(CachedColumn).init(allocator);
    defer cache.deinit();

    const ctx = ExprContext{
        .allocator = allocator,
        .column_cache = &cache,
    };

    // Create: 10 + 5
    var left = Expr{ .value = .{ .integer = 10 } };
    var right = Expr{ .value = .{ .integer = 5 } };
    const bin = .{
        .op = ast.BinaryOp.add,
        .left = &left,
        .right = &right,
    };

    const result = try evaluateBinaryToValue(ctx, bin, 0);
    try std.testing.expectEqual(@as(i64, 15), result.integer);
}

test "expr: inferExpressionType" {
    const allocator = std.testing.allocator;

    var cache = std.StringHashMap(CachedColumn).init(allocator);
    defer cache.deinit();

    const int_data = try allocator.alloc(i64, 1);
    defer allocator.free(int_data);
    int_data[0] = 42;
    try cache.put("count", CachedColumn{ .int64 = int_data });

    const float_data = try allocator.alloc(f64, 1);
    defer allocator.free(float_data);
    float_data[0] = 3.14;
    try cache.put("value", CachedColumn{ .float64 = float_data });

    const ctx = ExprContext{
        .allocator = allocator,
        .column_cache = &cache,
    };

    // Integer literal
    var int_expr = Expr{ .value = .{ .integer = 42 } };
    try std.testing.expectEqual(ResultType.int64, try inferExpressionType(ctx, &int_expr));

    // Float literal
    var float_expr = Expr{ .value = .{ .float = 3.14 } };
    try std.testing.expectEqual(ResultType.float64, try inferExpressionType(ctx, &float_expr));

    // Column reference
    var col_expr = Expr{ .column = .{ .name = "count", .table = null } };
    try std.testing.expectEqual(ResultType.int64, try inferExpressionType(ctx, &col_expr));

    var col_expr2 = Expr{ .column = .{ .name = "value", .table = null } };
    try std.testing.expectEqual(ResultType.float64, try inferExpressionType(ctx, &col_expr2));
}
