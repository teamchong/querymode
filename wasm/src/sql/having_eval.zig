//! HAVING Clause Evaluation - Standalone functions for HAVING clause evaluation
//!
//! This module contains pure functions for evaluating HAVING clauses on aggregated results.
//! All functions receive explicit parameters (composition pattern).

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;
const BinaryOp = ast.BinaryOp;
const Value = ast.Value;
const result_types = @import("result_types.zig");
const Result = result_types.Result;
pub const result_ops = @import("result_ops.zig");
pub const scalar_functions = @import("scalar_functions.zig");
pub const aggregate_functions = @import("aggregate_functions.zig");

/// Apply HAVING filter to a result set
/// Filters rows based on the HAVING expression and modifies the result in place
pub fn applyHaving(
    allocator: std.mem.Allocator,
    result: *Result,
    having_expr: *const Expr,
    select_items: []const ast.SelectItem,
) !void {
    if (result.row_count == 0) return;

    // Collect indices of rows that pass the HAVING filter
    var passing_indices = std.ArrayList(usize){};
    defer passing_indices.deinit(allocator);

    for (0..result.row_count) |row_idx| {
        const passes = try evaluateHavingExpr(result.columns, select_items, having_expr, row_idx);
        if (passes) {
            try passing_indices.append(allocator, row_idx);
        }
    }

    // If all rows pass, nothing to do
    if (passing_indices.items.len == result.row_count) return;

    // Build filtered result columns
    const indices = passing_indices.items;
    var new_columns = try allocator.alloc(Result.Column, result.columns.len);
    errdefer allocator.free(new_columns);

    for (result.columns, 0..) |col, i| {
        new_columns[i] = try result_ops.filterColumnByIndices(allocator, col, indices);
    }

    // Free old column data
    for (result.columns) |col| {
        col.data.free(allocator);
    }
    allocator.free(result.columns);

    result.columns = new_columns;
    result.row_count = indices.len;
}

/// Evaluate HAVING expression for a single result row
pub fn evaluateHavingExpr(
    columns: []const Result.Column,
    select_items: []const ast.SelectItem,
    expr: *const Expr,
    row_idx: usize,
) anyerror!bool {
    return switch (expr.*) {
        .value => |val| switch (val) {
            .integer => |i| i != 0,
            .float => |f| f != 0.0,
            .null => false,
            else => true,
        },
        .binary => |bin| try evaluateHavingBinaryOp(columns, select_items, bin.op, bin.left, bin.right, row_idx),
        .unary => |un| switch (un.op) {
            .not => !(try evaluateHavingExpr(columns, select_items, un.operand, row_idx)),
            else => error.UnsupportedOperator,
        },
        else => error.UnsupportedExpression,
    };
}

/// Evaluate binary operation in HAVING context
fn evaluateHavingBinaryOp(
    columns: []const Result.Column,
    select_items: []const ast.SelectItem,
    op: BinaryOp,
    left: *const Expr,
    right: *const Expr,
    row_idx: usize,
) anyerror!bool {
    return switch (op) {
        .@"and" => (try evaluateHavingExpr(columns, select_items, left, row_idx)) and
            (try evaluateHavingExpr(columns, select_items, right, row_idx)),
        .@"or" => (try evaluateHavingExpr(columns, select_items, left, row_idx)) or
            (try evaluateHavingExpr(columns, select_items, right, row_idx)),
        .eq, .ne, .lt, .le, .gt, .ge => try evaluateHavingComparison(columns, select_items, op, left, right, row_idx),
        else => error.UnsupportedOperator,
    };
}

/// Evaluate comparison in HAVING context
fn evaluateHavingComparison(
    columns: []const Result.Column,
    select_items: []const ast.SelectItem,
    op: BinaryOp,
    left: *const Expr,
    right: *const Expr,
    row_idx: usize,
) !bool {
    const left_val = try getHavingValue(columns, select_items, left, row_idx);
    const right_val = try getHavingValue(columns, select_items, right, row_idx);

    // Compare as floats for numeric comparison
    return switch (left_val) {
        .integer => |left_int| blk: {
            const right_num = switch (right_val) {
                .integer => |i| @as(f64, @floatFromInt(i)),
                .float => |f| f,
                else => return error.TypeMismatch,
            };
            const left_num = @as(f64, @floatFromInt(left_int));
            break :blk scalar_functions.compareNumbers(op, left_num, right_num);
        },
        .float => |left_float| blk: {
            const right_num = switch (right_val) {
                .integer => |i| @as(f64, @floatFromInt(i)),
                .float => |f| f,
                else => return error.TypeMismatch,
            };
            break :blk scalar_functions.compareNumbers(op, left_float, right_num);
        },
        .string => |left_str| blk: {
            const right_str = switch (right_val) {
                .string => |s| s,
                else => return error.TypeMismatch,
            };
            break :blk scalar_functions.compareStrings(op, left_str, right_str);
        },
        .null => op == .ne,
        else => error.UnsupportedType,
    };
}

/// Get value of expression in HAVING context (from result columns)
fn getHavingValue(
    columns: []const Result.Column,
    select_items: []const ast.SelectItem,
    expr: *const Expr,
    row_idx: usize,
) !Value {
    return switch (expr.*) {
        .value => expr.value,
        .column => |col| blk: {
            // Look up column by name in result columns
            const col_idx = result_ops.findColumnIndex(columns, col.name) orelse
                return error.ColumnNotFound;
            break :blk getResultColumnValue(columns[col_idx], row_idx);
        },
        .call => |call| blk: {
            // For aggregate functions, find matching SELECT item
            if (aggregate_functions.isAggregateFunction(call.name)) {
                const col_idx = try findAggregateColumnIndex(columns, select_items, call.name, call.args);
                break :blk getResultColumnValue(columns[col_idx], row_idx);
            }
            return error.UnsupportedExpression;
        },
        .binary => |bin| try evaluateHavingBinaryToValue(columns, select_items, bin, row_idx),
        else => error.UnsupportedExpression,
    };
}

/// Evaluate binary expression to value in HAVING context
fn evaluateHavingBinaryToValue(
    columns: []const Result.Column,
    select_items: []const ast.SelectItem,
    bin: anytype,
    row_idx: usize,
) anyerror!Value {
    const left = try getHavingValue(columns, select_items, bin.left, row_idx);
    const right = try getHavingValue(columns, select_items, bin.right, row_idx);

    return switch (bin.op) {
        .add => scalar_functions.addValues(left, right),
        .subtract => scalar_functions.subtractValues(left, right),
        .multiply => scalar_functions.multiplyValues(left, right),
        .divide => scalar_functions.divideValues(left, right),
        else => error.UnsupportedOperator,
    };
}

/// Get value from a result column at a given row index
pub fn getResultColumnValue(col: Result.Column, row_idx: usize) Value {
    return switch (col.data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| Value{ .integer = data[row_idx] },
        .int32, .date32 => |data| Value{ .integer = data[row_idx] },
        .float64 => |data| Value{ .float = data[row_idx] },
        .float32 => |data| Value{ .float = data[row_idx] },
        .bool_ => |data| Value{ .integer = if (data[row_idx]) 1 else 0 },
        .string => |data| Value{ .string = data[row_idx] },
    };
}

/// Find the result column index that matches an aggregate function call
pub fn findAggregateColumnIndex(
    columns: []const Result.Column,
    select_items: []const ast.SelectItem,
    call_name: []const u8,
    call_args: []const Expr,
) !usize {
    // First, try to find by alias matching the function name
    for (columns, 0..) |col, i| {
        if (std.ascii.eqlIgnoreCase(col.name, call_name)) {
            return i;
        }
    }

    // Match by comparing SELECT item expressions
    for (select_items, 0..) |item, i| {
        if (item.expr == .call) {
            const item_call = item.expr.call;
            // Match function name (case insensitive)
            if (std.ascii.eqlIgnoreCase(item_call.name, call_name)) {
                // Match arguments
                if (aggregate_functions.aggregateArgsMatch(item_call.args, call_args)) {
                    return i;
                }
            }
        }
    }

    return error.ColumnNotFound;
}

// ============================================================================
// Tests
// ============================================================================

test "having: getResultColumnValue" {
    const allocator = std.testing.allocator;

    // Create test columns
    const int_data = try allocator.alloc(i64, 3);
    defer allocator.free(int_data);
    int_data[0] = 10;
    int_data[1] = 20;
    int_data[2] = 30;

    const float_data = try allocator.alloc(f64, 3);
    defer allocator.free(float_data);
    float_data[0] = 1.5;
    float_data[1] = 2.5;
    float_data[2] = 3.5;

    const int_col = Result.Column{ .name = "count", .data = .{ .int64 = int_data } };
    const float_col = Result.Column{ .name = "avg", .data = .{ .float64 = float_data } };

    // Test integer column
    const val0 = getResultColumnValue(int_col, 0);
    try std.testing.expectEqual(@as(i64, 10), val0.integer);

    const val2 = getResultColumnValue(int_col, 2);
    try std.testing.expectEqual(@as(i64, 30), val2.integer);

    // Test float column
    const fval1 = getResultColumnValue(float_col, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), fval1.float, 0.001);
}
