//! GROUP BY Evaluation - Group aggregation and key hashing
//!
//! This module contains pure functions for GROUP BY evaluation.
//! Uses composition pattern with explicit context parameters.
//!
//! IMPORTANT: Columns must be pre-loaded into column_cache before calling these functions.

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;
const hash = @import("lanceql.hash");
const result_types = @import("result_types.zig");
const Result = result_types.Result;
const CachedColumn = result_types.CachedColumn;
pub const aggregate_functions = @import("aggregate_functions.zig");
const AggregateType = aggregate_functions.AggregateType;
const Accumulator = aggregate_functions.Accumulator;
const PercentileAccumulator = aggregate_functions.PercentileAccumulator;
const columnar_ops = @import("lanceql.columnar_ops");

/// Group evaluation context
pub const GroupContext = struct {
    allocator: std.mem.Allocator,
    column_cache: *std.StringHashMap(CachedColumn),
};

/// Hash a group key from GROUP BY column values (efficient integer hashing)
///
/// This is faster than string key building because:
/// 1. No string allocation per row
/// 2. O(1) hash comparison instead of O(n) string comparison
/// 3. No type conversion overhead
pub fn hashGroupKey(ctx: GroupContext, group_cols: []const []const u8, row_idx: u32) u64 {
    if (group_cols.len == 0) {
        // No GROUP BY - all rows in one group
        return 0;
    }

    var key_hash: u64 = hash.FNV_OFFSET_BASIS;

    for (group_cols) |col_name| {
        const cached = ctx.column_cache.get(col_name) orelse continue;

        const col_hash = switch (cached) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| hash.hashI64(data[row_idx]),
            .int32, .date32 => |data| hash.hashI32(data[row_idx]),
            .float64 => |data| hash.hashF64(data[row_idx]),
            .float32 => |data| hash.hashF32(data[row_idx]),
            .bool_ => |data| hash.hashBool(data[row_idx]),
            .string => |data| hash.stringHash(data[row_idx]),
        };

        key_hash = hash.combineHash(key_hash, col_hash);
    }

    return key_hash;
}

/// Build a group key string from GROUP BY column values for a row
pub fn buildGroupKey(ctx: GroupContext, group_cols: []const []const u8, row_idx: u32) ![]const u8 {
    if (group_cols.len == 0) {
        // No GROUP BY - all rows in one group
        return try ctx.allocator.dupe(u8, "__all__");
    }

    var key = std.ArrayList(u8){};
    errdefer key.deinit(ctx.allocator);

    for (group_cols, 0..) |col_name, i| {
        if (i > 0) try key.append(ctx.allocator, '|');

        const cached = ctx.column_cache.get(col_name) orelse return error.ColumnNotCached;

        switch (cached) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| {
                var buf: [64]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                try key.appendSlice(ctx.allocator, str);
            },
            .int32, .date32 => |data| {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                try key.appendSlice(ctx.allocator, str);
            },
            .float64 => |data| {
                var buf: [64]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                try key.appendSlice(ctx.allocator, str);
            },
            .float32 => |data| {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{data[row_idx]}) catch |err| return err;
                try key.appendSlice(ctx.allocator, str);
            },
            .bool_ => |data| {
                const str = if (data[row_idx]) "true" else "false";
                try key.appendSlice(ctx.allocator, str);
            },
            .string => |data| {
                try key.appendSlice(ctx.allocator, data[row_idx]);
            },
        }
    }

    return key.toOwnedSlice(ctx.allocator);
}

/// Extract column names from any expression (recursive)
pub fn extractExprColumnNames(allocator: std.mem.Allocator, expr: *const Expr, list: *std.ArrayList([]const u8)) anyerror!void {
    switch (expr.*) {
        .column => |col| {
            if (!std.mem.eql(u8, col.name, "*")) {
                try list.append(allocator, col.name);
            }
        },
        .binary => |bin| {
            try extractExprColumnNames(allocator, bin.left, list);
            try extractExprColumnNames(allocator, bin.right, list);
        },
        .unary => |un| {
            try extractExprColumnNames(allocator, un.operand, list);
        },
        .call => |call| {
            for (call.args) |*arg| {
                try extractExprColumnNames(allocator, arg, list);
            }
        },
        else => {},
    }
}

/// Evaluate a SELECT item for all groups
pub fn evaluateSelectItemForGroups(
    ctx: GroupContext,
    item: ast.SelectItem,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    group_cols: []const []const u8,
    num_groups: usize,
) !Result.Column {
    const expr = &item.expr;

    // Handle aggregate function
    if (expr.* == .call and aggregate_functions.isAggregateFunction(expr.call.name)) {
        return evaluateAggregateForGroups(ctx, item, groups, num_groups);
    }

    // Handle regular column (must be in GROUP BY)
    if (expr.* == .column) {
        const col_name = expr.column.name;

        // Verify column is in GROUP BY
        var in_group_by = false;
        for (group_cols) |gb_col| {
            if (std.mem.eql(u8, gb_col, col_name)) {
                in_group_by = true;
                break;
            }
        }
        if (!in_group_by and group_cols.len > 0) {
            return error.ColumnNotInGroupBy;
        }

        return evaluateGroupByColumnForGroups(ctx, item, groups, num_groups);
    }

    return error.UnsupportedExpression;
}

/// Evaluate an aggregate function for all groups
pub fn evaluateAggregateForGroups(
    ctx: GroupContext,
    item: ast.SelectItem,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    num_groups: usize,
) !Result.Column {
    const call = item.expr.call;
    const agg_type = aggregate_functions.parseAggregateTypeWithArgs(call.name, call.args);

    // Determine column name for the aggregate (if not COUNT(*))
    const agg_col_name: ?[]const u8 = if (agg_type != .count_star and call.args.len > 0)
        if (call.args[0] == .column) call.args[0].column.name else null
    else
        null;

    // Check if this is a percentile-based aggregate (requires storing all values)
    const is_percentile_agg = agg_type == .median or agg_type == .percentile;

    // Check if this is a float-returning aggregate (stddev, variance)
    const is_float_agg = agg_type == .stddev or agg_type == .stddev_pop or
        agg_type == .variance or agg_type == .var_pop or agg_type == .avg;

    if (is_percentile_agg) {
        return evaluatePercentileAggregate(ctx, item, groups, num_groups, agg_type, agg_col_name, call);
    } else if (is_float_agg) {
        return evaluateFloatAggregate(ctx, item, groups, num_groups, agg_type, agg_col_name);
    } else {
        return evaluateIntAggregate(ctx, item, groups, num_groups, agg_type, agg_col_name);
    }
}

/// Evaluate percentile-based aggregate (MEDIAN, PERCENTILE)
fn evaluatePercentileAggregate(
    ctx: GroupContext,
    item: ast.SelectItem,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    num_groups: usize,
    agg_type: AggregateType,
    agg_col_name: ?[]const u8,
    call: anytype,
) !Result.Column {
    const results = try ctx.allocator.alloc(f64, num_groups);
    errdefer ctx.allocator.free(results);

    // Handle case of no groups
    if (groups.count() == 0) {
        results[0] = 0;
        return Result.Column{
            .name = item.alias orelse call.name,
            .data = Result.ColumnData{ .float64 = results },
        };
    }

    // Get percentile value (0.5 for median, from second arg for percentile)
    const percentile_val: f64 = if (agg_type == .median)
        0.5
    else if (call.args.len >= 2 and call.args[1] == .value)
        switch (call.args[1].value) {
            .float => |f| f,
            .integer => |i| @as(f64, @floatFromInt(i)),
            else => 0.5,
        }
    else
        0.5;

    // Compute percentile for each group
    var group_idx: usize = 0;
    var iter = groups.iterator();
    while (iter.next()) |entry| {
        const row_indices = entry.value_ptr.items;

        var acc = PercentileAccumulator.init(ctx.allocator, percentile_val);
        defer acc.deinit();

        for (row_indices) |row_idx| {
            if (agg_col_name) |col_name| {
                const cached = ctx.column_cache.get(col_name) orelse return error.ColumnNotCached;
                try addValueToPercentileAcc(&acc, cached, row_idx);
            }
        }

        results[group_idx] = acc.getResult();
        group_idx += 1;
    }

    return Result.Column{
        .name = item.alias orelse call.name,
        .data = Result.ColumnData{ .float64 = results },
    };
}

/// Helper to add a cached column value to percentile accumulator
fn addValueToPercentileAcc(acc: *PercentileAccumulator, cached: CachedColumn, row_idx: u32) !void {
    switch (cached) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| try acc.addInt(data[row_idx]),
        .int32, .date32 => |data| try acc.addInt(data[row_idx]),
        .float64 => |data| try acc.addFloat(data[row_idx]),
        .float32 => |data| try acc.addFloat(data[row_idx]),
        .bool_ => |data| try acc.addInt(if (data[row_idx]) 1 else 0),
        .string => {}, // Skip strings for percentile
    }
}

/// Evaluate float-returning aggregate (AVG, STDDEV, VARIANCE)
fn evaluateFloatAggregate(
    ctx: GroupContext,
    item: ast.SelectItem,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    num_groups: usize,
    agg_type: AggregateType,
    agg_col_name: ?[]const u8,
) !Result.Column {
    const call = item.expr.call;
    const results = try ctx.allocator.alloc(f64, num_groups);
    errdefer ctx.allocator.free(results);

    // Handle case of no groups
    if (groups.count() == 0) {
        results[0] = 0;
        return Result.Column{
            .name = item.alias orelse call.name,
            .data = Result.ColumnData{ .float64 = results },
        };
    }

    // Compute aggregate for each group
    var group_idx: usize = 0;
    var iter = groups.iterator();
    while (iter.next()) |entry| {
        const row_indices = entry.value_ptr.items;

        // SIMD fast path for AVG on f64 columns
        if (agg_type == .avg and agg_col_name != null) {
            const col_name = agg_col_name.?;
            const cached = ctx.column_cache.get(col_name) orelse return error.ColumnNotCached;

            const simd_result = trySimdFloatAggregate(cached, row_indices, agg_type);
            if (simd_result) |result| {
                results[group_idx] = result;
                group_idx += 1;
                continue;
            }
        }

        // Fallback: row-by-row accumulation (for STDDEV, VARIANCE, or non-f64)
        var acc = Accumulator.init(agg_type);

        for (row_indices) |row_idx| {
            if (agg_col_name) |col_name| {
                const cached = ctx.column_cache.get(col_name) orelse return error.ColumnNotCached;
                addValueToAcc(&acc, cached, row_idx);
            } else {
                acc.addCount();
            }
        }

        results[group_idx] = acc.getResult();
        group_idx += 1;
    }

    return Result.Column{
        .name = item.alias orelse call.name,
        .data = Result.ColumnData{ .float64 = results },
    };
}

/// Try SIMD aggregate on f64 column, returns null if not applicable
fn trySimdFloatAggregate(cached: CachedColumn, row_indices: []const u32, agg_type: AggregateType) ?f64 {
    // Only handle f64 columns
    const data: []const f64 = switch (cached) {
        .float64 => |d| d,
        else => return null,
    };

    if (row_indices.len == 0) return 0;

    return switch (agg_type) {
        .avg => blk: {
            // AVG = SUM / COUNT using SIMD sum
            const sum = columnar_ops.sumFilteredF64(data, row_indices);
            break :blk sum / @as(f64, @floatFromInt(row_indices.len));
        },
        else => null, // STDDEV, VARIANCE need more complex accumulation
    };
}

/// Evaluate int-returning aggregate (COUNT, SUM, MIN, MAX)
fn evaluateIntAggregate(
    ctx: GroupContext,
    item: ast.SelectItem,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    num_groups: usize,
    agg_type: AggregateType,
    agg_col_name: ?[]const u8,
) !Result.Column {
    const call = item.expr.call;
    const results = try ctx.allocator.alloc(i64, num_groups);
    errdefer ctx.allocator.free(results);

    // Handle case of no groups
    if (groups.count() == 0) {
        results[0] = 0;
        return Result.Column{
            .name = item.alias orelse call.name,
            .data = Result.ColumnData{ .int64 = results },
        };
    }

    // Compute aggregate for each group
    var group_idx: usize = 0;
    var iter = groups.iterator();
    while (iter.next()) |entry| {
        const row_indices = entry.value_ptr.items;

        // SIMD fast path for simple aggregates on i64 columns
        if (agg_col_name) |col_name| {
            const cached = ctx.column_cache.get(col_name) orelse return error.ColumnNotCached;

            // Try SIMD path for i64 columns
            const simd_result = trySimdIntAggregate(cached, row_indices, agg_type);
            if (simd_result) |result| {
                results[group_idx] = result;
                group_idx += 1;
                continue;
            }
        }

        // Fallback: row-by-row accumulation
        var acc = Accumulator.init(agg_type);

        for (row_indices) |row_idx| {
            if (agg_type == .count_star) {
                acc.addCount();
            } else if (agg_col_name) |col_name| {
                const cached = ctx.column_cache.get(col_name) orelse return error.ColumnNotCached;
                addValueToAcc(&acc, cached, row_idx);
            } else {
                acc.addCount();
            }
        }

        results[group_idx] = acc.getIntResult();
        group_idx += 1;
    }

    return Result.Column{
        .name = item.alias orelse call.name,
        .data = Result.ColumnData{ .int64 = results },
    };
}

/// Try SIMD aggregate on i64 column, returns null if not applicable
fn trySimdIntAggregate(cached: CachedColumn, row_indices: []const u32, agg_type: AggregateType) ?i64 {
    // Only handle i64 columns (and timestamp/date variants which are stored as i64)
    const data: []const i64 = switch (cached) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |d| d,
        else => return null,
    };

    return switch (agg_type) {
        .sum => blk: {
            // Use filtered SIMD sum
            const sum128 = columnar_ops.sumFilteredI64(data, row_indices);
            // Clamp to i64 (overflow is possible but matches row-by-row behavior)
            break :blk @as(i64, @intCast(@min(@max(sum128, std.math.minInt(i64)), std.math.maxInt(i64))));
        },
        .min => columnar_ops.minFilteredI64(data, row_indices),
        .max => columnar_ops.maxFilteredI64(data, row_indices),
        .count, .count_star => @intCast(row_indices.len),
        else => null, // AVG, STDDEV, etc. need float path
    };
}

/// Helper to add a cached column value to accumulator
fn addValueToAcc(acc: *Accumulator, cached: CachedColumn, row_idx: u32) void {
    switch (cached) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| acc.addInt(data[row_idx]),
        .int32, .date32 => |data| acc.addInt(data[row_idx]),
        .float64 => |data| acc.addFloat(data[row_idx]),
        .float32 => |data| acc.addFloat(data[row_idx]),
        .bool_ => acc.addCount(),
        .string => acc.addCount(),
    }
}

/// Evaluate a GROUP BY column for all groups (return first value from each group)
pub fn evaluateGroupByColumnForGroups(
    ctx: GroupContext,
    item: ast.SelectItem,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    num_groups: usize,
) !Result.Column {
    const col_name = item.expr.column.name;
    const cached = ctx.column_cache.get(col_name) orelse return error.ColumnNotCached;

    // Use inline switch to reduce code duplication
    return switch (cached) {
        .int64 => |src| try extractFirstValues(ctx.allocator, i64, src, groups, num_groups, item.alias orelse col_name, .int64),
        .timestamp_s => |src| try extractFirstValues(ctx.allocator, i64, src, groups, num_groups, item.alias orelse col_name, .timestamp_s),
        .timestamp_ms => |src| try extractFirstValues(ctx.allocator, i64, src, groups, num_groups, item.alias orelse col_name, .timestamp_ms),
        .timestamp_us => |src| try extractFirstValues(ctx.allocator, i64, src, groups, num_groups, item.alias orelse col_name, .timestamp_us),
        .timestamp_ns => |src| try extractFirstValues(ctx.allocator, i64, src, groups, num_groups, item.alias orelse col_name, .timestamp_ns),
        .date64 => |src| try extractFirstValues(ctx.allocator, i64, src, groups, num_groups, item.alias orelse col_name, .date64),
        .int32 => |src| try extractFirstValues(ctx.allocator, i32, src, groups, num_groups, item.alias orelse col_name, .int32),
        .date32 => |src| try extractFirstValues(ctx.allocator, i32, src, groups, num_groups, item.alias orelse col_name, .date32),
        .float64 => |src| try extractFirstValues(ctx.allocator, f64, src, groups, num_groups, item.alias orelse col_name, .float64),
        .float32 => |src| try extractFirstValues(ctx.allocator, f32, src, groups, num_groups, item.alias orelse col_name, .float32),
        .bool_ => |src| try extractFirstValues(ctx.allocator, bool, src, groups, num_groups, item.alias orelse col_name, .bool_),
        .string => |src| try extractFirstStringValues(ctx.allocator, src, groups, num_groups, item.alias orelse col_name),
    };
}

/// Generic helper to extract first value from each group
fn extractFirstValues(
    allocator: std.mem.Allocator,
    comptime T: type,
    source_data: []const T,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    num_groups: usize,
    col_name: []const u8,
    comptime data_tag: std.meta.FieldEnum(Result.ColumnData),
) !Result.Column {
    const results = try allocator.alloc(T, num_groups);
    errdefer allocator.free(results);

    var group_idx: usize = 0;
    var iter = groups.iterator();
    while (iter.next()) |entry| {
        const row_indices = entry.value_ptr.items;
        if (row_indices.len > 0) {
            results[group_idx] = source_data[row_indices[0]];
        }
        group_idx += 1;
    }

    return Result.Column{
        .name = col_name,
        .data = @unionInit(Result.ColumnData, @tagName(data_tag), results),
    };
}

/// Extract first string value from each group (with duplication)
fn extractFirstStringValues(
    allocator: std.mem.Allocator,
    source_data: []const []const u8,
    groups: *std.AutoHashMap(u64, std.ArrayListUnmanaged(u32)),
    num_groups: usize,
    col_name: []const u8,
) !Result.Column {
    const results = try allocator.alloc([]const u8, num_groups);
    errdefer allocator.free(results);

    var group_idx: usize = 0;
    var iter = groups.iterator();
    while (iter.next()) |entry| {
        const row_indices = entry.value_ptr.items;
        if (row_indices.len > 0) {
            results[group_idx] = try allocator.dupe(u8, source_data[row_indices[0]]);
        }
        group_idx += 1;
    }

    return Result.Column{
        .name = col_name,
        .data = Result.ColumnData{ .string = results },
    };
}

// ============================================================================
// Tests
// ============================================================================

test "group: hashGroupKey single column" {
    const allocator = std.testing.allocator;

    var cache = std.StringHashMap(CachedColumn).init(allocator);
    defer cache.deinit();

    const data = try allocator.alloc(i64, 3);
    defer allocator.free(data);
    data[0] = 100;
    data[1] = 100;
    data[2] = 200;

    try cache.put("category", CachedColumn{ .int64 = data });

    const ctx = GroupContext{
        .allocator = allocator,
        .column_cache = &cache,
    };

    const group_cols = &[_][]const u8{"category"};

    // Same value should produce same hash
    const hash0 = hashGroupKey(ctx, group_cols, 0);
    const hash1 = hashGroupKey(ctx, group_cols, 1);
    const hash2 = hashGroupKey(ctx, group_cols, 2);

    try std.testing.expectEqual(hash0, hash1);
    try std.testing.expect(hash0 != hash2);
}

test "group: hashGroupKey no columns" {
    const allocator = std.testing.allocator;

    var cache = std.StringHashMap(CachedColumn).init(allocator);
    defer cache.deinit();

    const ctx = GroupContext{
        .allocator = allocator,
        .column_cache = &cache,
    };

    // No GROUP BY - all rows get same hash (0)
    const hash0 = hashGroupKey(ctx, &[_][]const u8{}, 0);
    const hash1 = hashGroupKey(ctx, &[_][]const u8{}, 1);

    try std.testing.expectEqual(@as(u64, 0), hash0);
    try std.testing.expectEqual(hash0, hash1);
}

test "group: extractExprColumnNames" {
    const allocator = std.testing.allocator;

    var list = std.ArrayList([]const u8){};
    defer list.deinit(allocator);

    // Test column expression
    var col_expr = Expr{ .column = .{ .name = "price", .table = null } };
    try extractExprColumnNames(allocator, &col_expr, &list);

    try std.testing.expectEqual(@as(usize, 1), list.items.len);
    try std.testing.expectEqualStrings("price", list.items[0]);
}
