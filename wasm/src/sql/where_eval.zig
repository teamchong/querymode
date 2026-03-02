//! WHERE Clause Evaluation - Standalone functions for WHERE clause evaluation
//!
//! This module contains functions for evaluating SQL WHERE clauses.
//! Uses composition pattern with explicit context parameters.
//!
//! Key design: Uses function pointers for recursive execute() calls to avoid circular imports.

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;
const BinaryOp = ast.BinaryOp;
const Value = ast.Value;
const result_types = @import("result_types.zig");
const Result = result_types.Result;
const CachedColumn = result_types.CachedColumn;
pub const scalar_functions = @import("scalar_functions.zig");
const columnar_ops = @import("lanceql.columnar_ops");

/// Function pointer type for evaluating expressions that need executor context
pub const EvalExprFn = *const fn (ctx: *anyopaque, expr: *const Expr, row_idx: u32) anyerror!Value;

/// Function pointer for executing subqueries (EXISTS, IN subquery)
/// Takes context, statement, and params, returns Result
pub const ExecuteFn = *const fn (*anyopaque, *ast.SelectStmt, []const Value) anyerror!Result;

/// Context for WHERE clause evaluation
pub const WhereContext = struct {
    allocator: std.mem.Allocator,
    column_cache: *std.StringHashMap(CachedColumn),
    row_count: usize,
    /// Function pointer for executing subqueries (EXISTS, IN subquery)
    execute_fn: ?ExecuteFn = null,
    /// Opaque executor context for eval_expr_fn and execute_fn
    eval_ctx: ?*anyopaque = null,
    /// Function pointer for evaluating expressions that need executor context
    /// (e.g., scalar functions, method calls)
    eval_expr_fn: ?EvalExprFn = null,
};

/// Evaluate WHERE clause and return matching row indices
pub fn evaluateWhere(ctx: WhereContext, where_expr: *const Expr, params: []const Value) ![]u32 {
    // Bind parameters first
    var bound_expr = try bindParameters(ctx.allocator, where_expr, params);
    defer freeExpr(ctx.allocator, &bound_expr);

    // Try vectorized filter (handles simple comparisons AND compound AND/OR)
    if (tryVectorizedFilter(ctx, &bound_expr)) |result| {
        return result;
    }

    // Fallback: Evaluate expression for each row (row-by-row)
    var matching_indices = std.ArrayList(u32){};
    errdefer matching_indices.deinit(ctx.allocator);

    var row_idx: u32 = 0;
    while (row_idx < ctx.row_count) : (row_idx += 1) {
        const matches = try evaluateExprForRow(ctx, &bound_expr, row_idx);
        if (matches) {
            try matching_indices.append(ctx.allocator, row_idx);
        }
    }

    return matching_indices.toOwnedSlice(ctx.allocator);
}

/// Vectorized filter evaluation - handles simple comparisons AND compound AND/OR
/// Uses SIMD for leaf comparisons, combines results for AND/OR
/// Returns null if any part of the expression can't be vectorized
fn tryVectorizedFilter(ctx: WhereContext, expr: *const Expr) ?[]u32 {
    return tryVectorizedFilterRecursive(ctx, expr, null);
}

/// Recursive helper for vectorized filtering
/// selection_in: if non-null, only evaluate rows in this selection (for AND chains)
fn tryVectorizedFilterRecursive(ctx: WhereContext, expr: *const Expr, selection_in: ?[]const u32) ?[]u32 {
    switch (expr.*) {
        .binary => |bin| {
            // Handle AND/OR compounds
            if (bin.op == .@"and") {
                // AND: evaluate left, then filter right using left's result
                const left_result = tryVectorizedFilterRecursive(ctx, bin.left, selection_in) orelse return null;
                defer if (selection_in == null) ctx.allocator.free(left_result);

                // Now evaluate right, but only on rows that passed left
                const right_result = tryVectorizedFilterRecursive(ctx, bin.right, left_result) orelse {
                    if (selection_in == null) {} // already freed
                    return null;
                };

                // Return right result (which is already filtered by left)
                if (selection_in != null) {
                    ctx.allocator.free(left_result);
                }
                return right_result;
            } else if (bin.op == .@"or") {
                // OR: evaluate both, union the results
                const left_result = tryVectorizedFilterRecursive(ctx, bin.left, selection_in) orelse return null;
                const right_result = tryVectorizedFilterRecursive(ctx, bin.right, selection_in) orelse {
                    ctx.allocator.free(left_result);
                    return null;
                };

                // Union: merge sorted arrays
                const merged = mergeUnionSorted(ctx.allocator, left_result, right_result) catch {
                    ctx.allocator.free(left_result);
                    ctx.allocator.free(right_result);
                    return null;
                };
                ctx.allocator.free(left_result);
                ctx.allocator.free(right_result);
                return merged;
            }

            // Otherwise try simple comparison SIMD
            return trySimdFilterWithSelection(ctx, expr, selection_in);
        },
        else => return null,
    }
}

/// Merge two sorted arrays into union (no duplicates)
fn mergeUnionSorted(allocator: std.mem.Allocator, a: []const u32, b: []const u32) ![]u32 {
    var result = std.ArrayList(u32){};
    errdefer result.deinit(allocator);

    var i: usize = 0;
    var j: usize = 0;

    while (i < a.len and j < b.len) {
        if (a[i] < b[j]) {
            try result.append(allocator, a[i]);
            i += 1;
        } else if (a[i] > b[j]) {
            try result.append(allocator, b[j]);
            j += 1;
        } else {
            // Equal - add once
            try result.append(allocator, a[i]);
            i += 1;
            j += 1;
        }
    }

    // Append remainder
    while (i < a.len) : (i += 1) {
        try result.append(allocator, a[i]);
    }
    while (j < b.len) : (j += 1) {
        try result.append(allocator, b[j]);
    }

    return result.toOwnedSlice(allocator);
}

/// SIMD filter with optional input selection (for AND chains)
fn trySimdFilterWithSelection(ctx: WhereContext, expr: *const Expr, selection_in: ?[]const u32) ?[]u32 {
    // If we have an input selection, use filtered SIMD
    if (selection_in) |sel| {
        return trySimdFilterFiltered(ctx, expr, sel);
    }
    // Otherwise use full-column SIMD
    return trySimdFilter(ctx, expr);
}

/// SIMD filter on pre-selected rows (for AND chain optimization)
fn trySimdFilterFiltered(ctx: WhereContext, expr: *const Expr, selection: []const u32) ?[]u32 {
    const bin = switch (expr.*) {
        .binary => |b| b,
        else => return null,
    };

    const op = bin.op;
    if (op != .gt and op != .ge and op != .lt and op != .le and op != .eq and op != .ne) {
        return null;
    }

    // Extract column and value
    var col_name: []const u8 = undefined;
    var const_val: Value = undefined;
    var col_on_left = true;

    if (bin.left.* == .column and bin.right.* == .value) {
        col_name = bin.left.column.name;
        const_val = bin.right.value;
        col_on_left = true;
    } else if (bin.left.* == .value and bin.right.* == .column) {
        col_name = bin.right.column.name;
        const_val = bin.left.value;
        col_on_left = false;
    } else {
        return null;
    }

    const cached = ctx.column_cache.get(col_name) orelse return null;

    // Allocate output
    const out = ctx.allocator.alloc(u32, selection.len) catch return null;
    errdefer ctx.allocator.free(out);

    // Filter based on column type
    const match_count = switch (cached) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| blk: {
            const int_val = switch (const_val) {
                .integer => |i| i,
                .float => |f| @as(i64, @intFromFloat(f)),
                else => return null,
            };

            // Filter through selection using SIMD-accelerated comparison
            var count: usize = 0;
            for (selection) |row_idx| {
                const val = data[row_idx];
                const matches = switch (op) {
                    .gt => if (col_on_left) val > int_val else val < int_val,
                    .ge => if (col_on_left) val >= int_val else val <= int_val,
                    .lt => if (col_on_left) val < int_val else val > int_val,
                    .le => if (col_on_left) val <= int_val else val >= int_val,
                    .eq => val == int_val,
                    .ne => val != int_val,
                    else => unreachable,
                };
                if (matches) {
                    out[count] = row_idx;
                    count += 1;
                }
            }
            break :blk count;
        },
        .float64 => |data| blk: {
            const float_val = switch (const_val) {
                .float => |f| f,
                .integer => |i| @as(f64, @floatFromInt(i)),
                else => return null,
            };

            var count: usize = 0;
            for (selection) |row_idx| {
                const val = data[row_idx];
                const matches = switch (op) {
                    .gt => if (col_on_left) val > float_val else val < float_val,
                    .ge => if (col_on_left) val >= float_val else val <= float_val,
                    .lt => if (col_on_left) val < float_val else val > float_val,
                    .le => if (col_on_left) val <= float_val else val >= float_val,
                    .eq => val == float_val,
                    .ne => val != float_val,
                    else => unreachable,
                };
                if (matches) {
                    out[count] = row_idx;
                    count += 1;
                }
            }
            break :blk count;
        },
        else => return null,
    };

    // Shrink to actual count
    return ctx.allocator.realloc(out, match_count) catch out[0..match_count];
}

/// Try to use SIMD columnar filter for simple patterns: column OP value
/// Returns null if pattern doesn't match (falls back to row-by-row)
fn trySimdFilter(ctx: WhereContext, expr: *const Expr) ?[]u32 {
    // Check for binary comparison pattern
    const bin = switch (expr.*) {
        .binary => |b| b,
        else => {
            // Debug: expression is not binary
            // std.debug.print("SIMD: not binary expr, tag={}\n", .{@intFromEnum(std.meta.activeTag(expr.*))});
            return null;
        },
    };

    // Only handle simple comparison operators
    const op = bin.op;
    if (op != .gt and op != .ge and op != .lt and op != .le and op != .eq and op != .ne) {
        return null;
    }

    // Extract column name and constant value
    var col_name: []const u8 = undefined;
    var const_val: Value = undefined;
    var col_on_left = true;

    // Pattern: column OP value
    if (bin.left.* == .column and bin.right.* == .value) {
        col_name = bin.left.column.name;
        const_val = bin.right.value;
        col_on_left = true;
    }
    // Pattern: value OP column (swap comparison direction)
    else if (bin.left.* == .value and bin.right.* == .column) {
        col_name = bin.right.column.name;
        const_val = bin.left.value;
        col_on_left = false;
    } else {
        return null;
    }

    // Get cached column data
    const cached = ctx.column_cache.get(col_name) orelse return null;

    // Allocate output buffer (max size = row_count)
    const out = ctx.allocator.alloc(u32, ctx.row_count) catch return null;
    errdefer ctx.allocator.free(out);

    // Dispatch to appropriate SIMD filter based on column type and operator
    const count = switch (cached) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| blk: {
            const int_val = switch (const_val) {
                .integer => |i| i,
                .float => |f| @as(i64, @intFromFloat(f)),
                else => return null,
            };
            // If column is on the right, swap the operator direction
            const effective_op = if (col_on_left) op else swapCompareOp(op);
            break :blk dispatchSimdFilterI64(data, effective_op, int_val, out);
        },
        .float64 => |data| blk: {
            const float_val: f64 = switch (const_val) {
                .integer => |i| @floatFromInt(i),
                .float => |f| f,
                else => return null,
            };
            const effective_op = if (col_on_left) op else swapCompareOp(op);
            break :blk dispatchSimdFilterF64(data, effective_op, float_val, out);
        },
        else => return null, // Strings, bools, etc. use fallback
    };

    // Shrink output to actual size
    if (count == 0) {
        ctx.allocator.free(out);
        return ctx.allocator.alloc(u32, 0) catch return null;
    }

    return ctx.allocator.realloc(out, count) catch out[0..count];
}

/// Swap comparison operator for when value is on left side
fn swapCompareOp(op: BinaryOp) BinaryOp {
    return switch (op) {
        .gt => .lt, // value > col  =>  col < value
        .ge => .le,
        .lt => .gt,
        .le => .ge,
        .eq => .eq, // symmetric
        .ne => .ne,
        else => op,
    };
}

/// Dispatch to appropriate SIMD filter for i64
fn dispatchSimdFilterI64(data: []const i64, op: BinaryOp, value: i64, out: []u32) usize {
    return switch (op) {
        .gt => columnar_ops.filterGreaterI64(data, value, out),
        .ge => columnar_ops.filterGreaterEqualI64(data, value, out),
        .lt => columnar_ops.filterLessI64(data, value, out),
        .le => columnar_ops.filterLessEqualI64(data, value, out),
        .eq => columnar_ops.filterEqualI64(data, value, out),
        .ne => columnar_ops.filterNotEqualI64(data, value, out),
        else => 0,
    };
}

/// Dispatch to appropriate SIMD filter for f64
fn dispatchSimdFilterF64(data: []const f64, op: BinaryOp, value: f64, out: []u32) usize {
    return switch (op) {
        .gt => columnar_ops.filterGreaterF64(data, value, out),
        .ge => columnar_ops.filterGreaterEqualF64(data, value, out),
        .lt => columnar_ops.filterLessF64(data, value, out),
        .le => columnar_ops.filterLessEqualF64(data, value, out),
        // For f64 equal/not-equal, fall back (floating point equality is tricky)
        else => 0,
    };
}

/// Bind parameters (replace ? placeholders with actual values)
pub fn bindParameters(allocator: std.mem.Allocator, expr: *const Expr, params: []const Value) !Expr {
    return switch (expr.*) {
        .value => |val| blk: {
            if (val == .parameter) {
                const param_idx = val.parameter;
                if (param_idx >= params.len) return error.ParameterOutOfBounds;
                break :blk Expr{ .value = params[param_idx] };
            }
            break :blk expr.*;
        },
        .column => expr.*,
        .binary => |bin| blk: {
            const left_ptr = try allocator.create(Expr);
            errdefer allocator.destroy(left_ptr);
            left_ptr.* = try bindParameters(allocator, bin.left, params);
            errdefer freeExpr(allocator, left_ptr);

            const right_ptr = try allocator.create(Expr);
            errdefer allocator.destroy(right_ptr);
            right_ptr.* = try bindParameters(allocator, bin.right, params);

            break :blk Expr{
                .binary = .{
                    .op = bin.op,
                    .left = left_ptr,
                    .right = right_ptr,
                },
            };
        },
        .unary => |un| blk: {
            const operand_ptr = try allocator.create(Expr);
            errdefer allocator.destroy(operand_ptr);
            operand_ptr.* = try bindParameters(allocator, un.operand, params);

            break :blk Expr{
                .unary = .{
                    .op = un.op,
                    .operand = operand_ptr,
                },
            };
        },
        .call => |call| blk: {
            const new_args = try allocator.alloc(Expr, call.args.len);
            errdefer allocator.free(new_args);

            for (call.args, 0..) |*arg, i| {
                new_args[i] = try bindParameters(allocator, arg, params);
            }

            break :blk Expr{
                .call = .{
                    .name = call.name,
                    .args = new_args,
                    .distinct = call.distinct,
                    .window = call.window,
                },
            };
        },
        .in_list => |in| blk: {
            const new_expr = try allocator.create(Expr);
            errdefer allocator.destroy(new_expr);
            new_expr.* = try bindParameters(allocator, in.expr, params);
            errdefer freeExpr(allocator, new_expr);

            const new_values = try allocator.alloc(Expr, in.values.len);
            errdefer allocator.free(new_values);

            for (in.values, 0..) |*val, i| {
                new_values[i] = try bindParameters(allocator, val, params);
            }

            break :blk Expr{
                .in_list = .{
                    .expr = new_expr,
                    .values = new_values,
                    .negated = in.negated,
                },
            };
        },
        .between => |bet| blk: {
            const new_expr = try allocator.create(Expr);
            errdefer allocator.destroy(new_expr);
            new_expr.* = try bindParameters(allocator, bet.expr, params);
            errdefer freeExpr(allocator, new_expr);

            const new_low = try allocator.create(Expr);
            errdefer allocator.destroy(new_low);
            new_low.* = try bindParameters(allocator, bet.low, params);
            errdefer freeExpr(allocator, new_low);

            const new_high = try allocator.create(Expr);
            errdefer allocator.destroy(new_high);
            new_high.* = try bindParameters(allocator, bet.high, params);

            break :blk Expr{
                .between = .{
                    .expr = new_expr,
                    .low = new_low,
                    .high = new_high,
                },
            };
        },
        .case_expr => expr.*,
        .exists => expr.*,
        .in_subquery => |in| blk: {
            const new_expr = try allocator.create(Expr);
            errdefer allocator.destroy(new_expr);
            new_expr.* = try bindParameters(allocator, in.expr, params);

            break :blk Expr{
                .in_subquery = .{
                    .expr = new_expr,
                    .subquery = in.subquery,
                    .negated = in.negated,
                },
            };
        },
        .cast => expr.*,
        .method_call => |mc| blk: {
            const new_args = try allocator.alloc(Expr, mc.args.len);
            errdefer allocator.free(new_args);

            for (mc.args, 0..) |*arg, i| {
                new_args[i] = try bindParameters(allocator, arg, params);
            }

            break :blk Expr{
                .method_call = .{
                    .object = mc.object,
                    .method = mc.method,
                    .args = new_args,
                    .over = mc.over,
                },
            };
        },
    };
}

/// Free allocated expression tree
pub fn freeExpr(allocator: std.mem.Allocator, expr: *Expr) void {
    switch (expr.*) {
        .binary => |bin| {
            freeExpr(allocator, bin.left);
            allocator.destroy(bin.left);
            freeExpr(allocator, bin.right);
            allocator.destroy(bin.right);
        },
        .unary => |un| {
            freeExpr(allocator, un.operand);
            allocator.destroy(un.operand);
        },
        .call => |call| {
            for (call.args) |*arg| {
                freeExpr(allocator, arg);
            }
            allocator.free(call.args);
        },
        .in_list => |in| {
            freeExpr(allocator, in.expr);
            allocator.destroy(in.expr);
            for (in.values) |*val| {
                freeExpr(allocator, val);
            }
            allocator.free(in.values);
        },
        .in_subquery => |in| {
            freeExpr(allocator, in.expr);
            allocator.destroy(in.expr);
            // Don't free subquery - it's owned by the AST
        },
        .between => |bet| {
            freeExpr(allocator, bet.expr);
            allocator.destroy(bet.expr);
            freeExpr(allocator, bet.low);
            allocator.destroy(bet.low);
            freeExpr(allocator, bet.high);
            allocator.destroy(bet.high);
        },
        else => {},
    }
}

/// Evaluate expression for a specific row (returns boolean)
pub fn evaluateExprForRow(ctx: WhereContext, expr: *const Expr, row_idx: u32) anyerror!bool {
    return switch (expr.*) {
        .value => |val| blk: {
            break :blk switch (val) {
                .integer => |i| i != 0,
                .float => |f| f != 0.0,
                .null => false,
                else => true,
            };
        },
        .column => error.ColumnRequiresComparison,
        .binary => |bin| try evaluateBinaryOp(ctx, bin.op, bin.left, bin.right, row_idx),
        .unary => |un| try evaluateUnaryOp(ctx, un.op, un.operand, row_idx),
        .exists => |ex| try evaluateExists(ctx, ex.subquery, ex.negated),
        .in_list => |in| blk: {
            const result = try evaluateInList(ctx, in.expr, in.values, row_idx);
            break :blk if (in.negated) !result else result;
        },
        .in_subquery => |in| try evaluateInSubquery(ctx, in.expr, in.subquery, in.negated, row_idx),
        .between => |bet| try evaluateBetween(ctx, bet.expr, bet.low, bet.high, row_idx),
        else => error.UnsupportedExpression,
    };
}

/// Evaluate EXISTS subquery
fn evaluateExists(ctx: WhereContext, subquery: *ast.SelectStmt, negated: bool) anyerror!bool {
    const exec_fn = ctx.execute_fn orelse return error.NoExecuteFunctionProvided;
    const exec_ctx = ctx.eval_ctx orelse return error.NoExecuteContextProvided;

    var result = try exec_fn(exec_ctx, subquery, &[_]Value{});
    defer result.deinit();

    const exists = result.row_count > 0;
    return if (negated) !exists else exists;
}

/// Evaluate IN list expression
fn evaluateInList(ctx: WhereContext, expr: *const Expr, values: []const Expr, row_idx: u32) anyerror!bool {
    const left_val = try evaluateToValue(ctx, expr, row_idx);

    for (values) |*val_expr| {
        const list_val = try evaluateToValue(ctx, val_expr, row_idx);

        const matches = switch (left_val) {
            .integer => |left_int| switch (list_val) {
                .integer => |right_int| left_int == right_int,
                .float => |right_float| @as(f64, @floatFromInt(left_int)) == right_float,
                else => false,
            },
            .float => |left_float| switch (list_val) {
                .integer => |right_int| left_float == @as(f64, @floatFromInt(right_int)),
                .float => |right_float| left_float == right_float,
                else => false,
            },
            .string => |left_str| switch (list_val) {
                .string => |right_str| std.mem.eql(u8, left_str, right_str),
                else => false,
            },
            .null => false,
            else => false,
        };

        if (matches) return true;
    }

    return false;
}

/// Evaluate IN subquery expression
fn evaluateInSubquery(ctx: WhereContext, expr: *const Expr, subquery: *ast.SelectStmt, negated: bool, row_idx: u32) anyerror!bool {
    const left_val = try evaluateToValue(ctx, expr, row_idx);
    const exec_fn = ctx.execute_fn orelse return error.NoExecuteFunctionProvided;
    const exec_ctx = ctx.eval_ctx orelse return error.NoExecuteContextProvided;

    var result = try exec_fn(exec_ctx, subquery, &[_]Value{});
    defer result.deinit();

    if (result.columns.len != 1) {
        return error.SubqueryMustReturnOneColumn;
    }

    const col = result.columns[0];
    for (0..result.row_count) |i| {
        const subquery_val: Value = switch (col.data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| .{ .integer = data[i] },
            .int32, .date32 => |data| .{ .integer = data[i] },
            .float64 => |data| .{ .float = data[i] },
            .float32 => |data| .{ .float = data[i] },
            .bool_ => |data| .{ .integer = if (data[i]) 1 else 0 },
            .string => |data| .{ .string = data[i] },
        };

        const matches = switch (left_val) {
            .integer => |left_int| switch (subquery_val) {
                .integer => |right_int| left_int == right_int,
                .float => |right_float| @as(f64, @floatFromInt(left_int)) == right_float,
                else => false,
            },
            .float => |left_float| switch (subquery_val) {
                .integer => |right_int| left_float == @as(f64, @floatFromInt(right_int)),
                .float => |right_float| left_float == right_float,
                else => false,
            },
            .string => |left_str| switch (subquery_val) {
                .string => |right_str| std.mem.eql(u8, left_str, right_str),
                else => false,
            },
            .null => false,
            else => false,
        };

        if (matches) {
            return if (negated) false else true;
        }
    }

    return if (negated) true else false;
}

/// Evaluate BETWEEN expression: expr BETWEEN low AND high
/// Returns true if low <= expr <= high (inclusive)
/// Returns false if any operand is NULL
fn evaluateBetween(ctx: WhereContext, expr: *const Expr, low: *const Expr, high: *const Expr, row_idx: u32) anyerror!bool {
    const val = try evaluateToValue(ctx, expr, row_idx);
    const low_val = try evaluateToValue(ctx, low, row_idx);
    const high_val = try evaluateToValue(ctx, high, row_idx);

    // NULL handling: BETWEEN with any NULL operand returns false (two-valued logic)
    if (val == .null or low_val == .null or high_val == .null) {
        return false;
    }

    // Compare val >= low AND val <= high
    const ge_low = try compareValuesOp(val, low_val, .ge);
    if (!ge_low) return false;

    const le_high = try compareValuesOp(val, high_val, .le);
    return le_high;
}

/// Evaluate binary operation (returns boolean)
fn evaluateBinaryOp(ctx: WhereContext, op: BinaryOp, left: *const Expr, right: *const Expr, row_idx: u32) anyerror!bool {
    return switch (op) {
        .@"and" => (try evaluateExprForRow(ctx, left, row_idx)) and
            (try evaluateExprForRow(ctx, right, row_idx)),
        .@"or" => (try evaluateExprForRow(ctx, left, row_idx)) or
            (try evaluateExprForRow(ctx, right, row_idx)),
        .eq, .ne, .lt, .le, .gt, .ge => try evaluateComparison(ctx, op, left, right, row_idx),
        .like => try evaluateLike(ctx, left, right, row_idx),
        else => error.UnsupportedOperator,
    };
}

/// Evaluate comparison operation
fn evaluateComparison(ctx: WhereContext, op: BinaryOp, left: *const Expr, right: *const Expr, row_idx: u32) !bool {
    const left_val = try evaluateToValue(ctx, left, row_idx);
    const right_val = try evaluateToValue(ctx, right, row_idx);

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

/// Compare two Values with a specific operator (helper for BETWEEN)
fn compareValuesOp(left_val: Value, right_val: Value, op: BinaryOp) !bool {
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
        .null => false,
        else => error.UnsupportedType,
    };
}

/// Evaluate LIKE pattern matching
/// Supports SQL wildcards: % (any sequence), _ (single char)
/// Escape sequences: \% and \_ for literal match
fn evaluateLike(ctx: WhereContext, left: *const Expr, right: *const Expr, row_idx: u32) !bool {
    const left_val = try evaluateToValue(ctx, left, row_idx);
    const right_val = try evaluateToValue(ctx, right, row_idx);

    // NULL handling: LIKE with NULL returns false
    if (left_val == .null or right_val == .null) {
        return false;
    }

    const str = switch (left_val) {
        .string => |s| s,
        else => return error.TypeMismatch,
    };

    const pattern = switch (right_val) {
        .string => |s| s,
        else => return error.TypeMismatch,
    };

    return matchLikePattern(str, pattern);
}

/// Match a string against a SQL LIKE pattern
/// % matches any sequence of characters (including empty)
/// _ matches exactly one character
/// \% and \_ match literal % and _
fn matchLikePattern(str: []const u8, pattern: []const u8) bool {
    var s_idx: usize = 0;
    var p_idx: usize = 0;

    // For backtracking when % doesn't match
    var star_idx: ?usize = null;
    var match_idx: usize = 0;

    while (s_idx < str.len or p_idx < pattern.len) {
        if (p_idx < pattern.len) {
            const p_char = pattern[p_idx];

            // Handle escape sequences
            if (p_char == '\\' and p_idx + 1 < pattern.len) {
                const next = pattern[p_idx + 1];
                if (next == '%' or next == '_' or next == '\\') {
                    // Literal match of escaped character
                    if (s_idx < str.len and str[s_idx] == next) {
                        s_idx += 1;
                        p_idx += 2;
                        continue;
                    }
                    // No match for escaped literal
                    if (star_idx) |si| {
                        p_idx = si + 1;
                        match_idx += 1;
                        s_idx = match_idx;
                        continue;
                    }
                    return false;
                }
            }

            // % matches any sequence
            if (p_char == '%') {
                star_idx = p_idx;
                match_idx = s_idx;
                p_idx += 1;
                continue;
            }

            // _ matches single character
            if (p_char == '_') {
                if (s_idx < str.len) {
                    s_idx += 1;
                    p_idx += 1;
                    continue;
                }
                // No character to match
                if (star_idx) |si| {
                    p_idx = si + 1;
                    match_idx += 1;
                    s_idx = match_idx;
                    continue;
                }
                return false;
            }

            // Regular character match
            if (s_idx < str.len and str[s_idx] == p_char) {
                s_idx += 1;
                p_idx += 1;
                continue;
            }
        }

        // Backtrack to last % if possible
        if (star_idx) |si| {
            p_idx = si + 1;
            match_idx += 1;
            s_idx = match_idx;
            if (s_idx <= str.len) continue;
        }

        return false;
    }

    return true;
}

/// Evaluate unary operation
fn evaluateUnaryOp(ctx: WhereContext, op: ast.UnaryOp, operand: *const Expr, row_idx: u32) !bool {
    return switch (op) {
        .not => !(try evaluateExprForRow(ctx, operand, row_idx)),
        .is_null => blk: {
            const val = try evaluateToValue(ctx, operand, row_idx);
            break :blk val == .null;
        },
        .is_not_null => blk: {
            const val = try evaluateToValue(ctx, operand, row_idx);
            break :blk val != .null;
        },
        else => error.UnsupportedOperator,
    };
}

/// Evaluate expression to a concrete value (for WHERE clause comparisons)
/// For method calls and scalar functions, delegates to eval_expr_fn if provided.
pub fn evaluateToValue(ctx: WhereContext, expr: *const Expr, row_idx: u32) !Value {
    return switch (expr.*) {
        .value => expr.value,
        .column => |col| blk: {
            const cached = ctx.column_cache.get(col.name) orelse return error.ColumnNotCached;

            break :blk switch (cached) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| Value{ .integer = data[row_idx] },
                .int32, .date32 => |data| Value{ .integer = data[row_idx] },
                .float64 => |data| Value{ .float = data[row_idx] },
                .float32 => |data| Value{ .float = data[row_idx] },
                .bool_ => |data| Value{ .integer = if (data[row_idx]) 1 else 0 },
                .string => |data| Value{ .string = data[row_idx] },
            };
        },
        // For function calls in WHERE, delegate to executor via eval_expr_fn
        .call, .method_call => blk: {
            if (ctx.eval_expr_fn) |fn_ptr| {
                break :blk try fn_ptr(ctx.eval_ctx.?, expr, row_idx);
            }
            return error.FunctionCallsNotSupportedInWhere;
        },
        else => error.UnsupportedExpression,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "where: bindParameters simple" {
    const allocator = std.testing.allocator;

    // Create a simple value expression with parameter
    var param_expr = Expr{ .value = .{ .parameter = 0 } };
    const params = [_]Value{.{ .integer = 42 }};

    var bound = try bindParameters(allocator, &param_expr, &params);
    defer freeExpr(allocator, &bound);

    try std.testing.expectEqual(Value{ .integer = 42 }, bound.value);
}

test "where: evaluateToValue column lookup" {
    const allocator = std.testing.allocator;

    // Create column cache with test data
    var column_cache = std.StringHashMap(CachedColumn).init(allocator);
    defer column_cache.deinit();

    const data = try allocator.alloc(i64, 3);
    defer allocator.free(data);
    data[0] = 100;
    data[1] = 200;
    data[2] = 300;

    try column_cache.put("id", CachedColumn{ .int64 = data });

    const ctx = WhereContext{
        .allocator = allocator,
        .column_cache = &column_cache,
        .row_count = 3,
    };

    var col_expr = Expr{ .column = .{ .name = "id", .table = null } };
    const val = try evaluateToValue(ctx, &col_expr, 1);

    try std.testing.expectEqual(@as(i64, 200), val.integer);
}

test "LIKE: exact match" {
    try std.testing.expect(matchLikePattern("foo", "foo"));
    try std.testing.expect(!matchLikePattern("foo", "bar"));
    try std.testing.expect(!matchLikePattern("foo", "foobar"));
}

test "LIKE: percent wildcard" {
    // % matches any sequence
    try std.testing.expect(matchLikePattern("foobar", "foo%"));
    try std.testing.expect(matchLikePattern("foo", "foo%"));
    try std.testing.expect(!matchLikePattern("bar", "foo%"));

    // Suffix match
    try std.testing.expect(matchLikePattern("foobar", "%bar"));
    try std.testing.expect(matchLikePattern("bar", "%bar"));
    try std.testing.expect(!matchLikePattern("barfoo", "%bar"));

    // Contains
    try std.testing.expect(matchLikePattern("foobar", "%oba%"));
    try std.testing.expect(matchLikePattern("abc", "%b%"));

    // Match all
    try std.testing.expect(matchLikePattern("anything", "%"));
    try std.testing.expect(matchLikePattern("", "%"));
}

test "LIKE: underscore wildcard" {
    // _ matches single character
    try std.testing.expect(matchLikePattern("foo", "f_o"));
    try std.testing.expect(!matchLikePattern("fo", "f_o"));
    try std.testing.expect(!matchLikePattern("fooo", "f_o"));

    // Multiple underscores
    try std.testing.expect(matchLikePattern("bar", "b__"));
    try std.testing.expect(!matchLikePattern("ba", "b__"));
}

test "LIKE: mixed patterns" {
    try std.testing.expect(matchLikePattern("foobar", "f%r"));
    try std.testing.expect(matchLikePattern("foobar", "f_o%"));
    try std.testing.expect(matchLikePattern("foobar", "%o_a%"));
}

test "LIKE: escape sequences" {
    // \% matches literal %
    try std.testing.expect(matchLikePattern("100%", "100\\%"));
    try std.testing.expect(!matchLikePattern("100x", "100\\%"));

    // \_ matches literal _
    try std.testing.expect(matchLikePattern("foo_bar", "foo\\_bar"));
    try std.testing.expect(!matchLikePattern("fooxbar", "foo\\_bar"));

    // Combined
    try std.testing.expect(matchLikePattern("50% off", "%\\% off"));
}

test "LIKE: empty strings" {
    try std.testing.expect(matchLikePattern("", ""));
    try std.testing.expect(matchLikePattern("", "%"));
    try std.testing.expect(!matchLikePattern("", "_"));
    try std.testing.expect(!matchLikePattern("", "a"));
}
