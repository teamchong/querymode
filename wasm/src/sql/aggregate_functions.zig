//! Aggregate Functions - Types and utilities for SQL aggregate operations
//!
//! Contains AggregateType enum, accumulators, and aggregate detection functions.

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;

/// Aggregate function types
pub const AggregateType = enum {
    count,
    count_star,
    sum,
    avg,
    min,
    max,
    // Statistical aggregates
    stddev, // Sample standard deviation
    variance, // Sample variance
    stddev_pop, // Population standard deviation
    var_pop, // Population variance
    // Percentile-based aggregates (require storing all values)
    median, // 50th percentile
    percentile, // Arbitrary percentile (0-1)
};

/// Accumulator for aggregate computations
pub const Accumulator = struct {
    agg_type: AggregateType,
    count: i64,
    sum: f64,
    sum_sq: f64, // Sum of squares for variance/stddev
    min_int: ?i64,
    max_int: ?i64,
    min_float: ?f64,
    max_float: ?f64,

    pub fn init(agg_type: AggregateType) Accumulator {
        return Accumulator{
            .agg_type = agg_type,
            .count = 0,
            .sum = 0,
            .sum_sq = 0,
            .min_int = null,
            .max_int = null,
            .min_float = null,
            .max_float = null,
        };
    }

    pub fn addInt(self: *Accumulator, value: i64) void {
        const fval = @as(f64, @floatFromInt(value));
        self.count += 1;
        self.sum += fval;
        self.sum_sq += fval * fval;
        if (self.min_int == null or value < self.min_int.?) {
            self.min_int = value;
        }
        if (self.max_int == null or value > self.max_int.?) {
            self.max_int = value;
        }
    }

    pub fn addFloat(self: *Accumulator, value: f64) void {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        if (self.min_float == null or value < self.min_float.?) {
            self.min_float = value;
        }
        if (self.max_float == null or value > self.max_float.?) {
            self.max_float = value;
        }
    }

    pub fn addCount(self: *Accumulator) void {
        self.count += 1;
    }

    /// Compute variance using the formula: (sum_sq - sum²/n) / divisor
    /// where divisor is (n-1) for sample variance, n for population variance
    fn computeVariance(self: Accumulator, population: bool) f64 {
        if (self.count == 0) return 0;
        if (!population and self.count == 1) return 0; // Sample variance undefined for n=1
        const n = @as(f64, @floatFromInt(self.count));
        const mean = self.sum / n;
        // Variance = E[X²] - E[X]² = sum_sq/n - mean²
        const variance_pop = self.sum_sq / n - mean * mean;
        if (population) {
            return variance_pop;
        } else {
            // Sample variance: multiply by n/(n-1) to get unbiased estimate
            return variance_pop * n / (n - 1);
        }
    }

    pub fn getResult(self: Accumulator) f64 {
        return switch (self.agg_type) {
            .count, .count_star => @as(f64, @floatFromInt(self.count)),
            .sum => self.sum,
            .avg => if (self.count > 0) self.sum / @as(f64, @floatFromInt(self.count)) else 0,
            .min => self.min_float orelse @as(f64, @floatFromInt(self.min_int orelse 0)),
            .max => self.max_float orelse @as(f64, @floatFromInt(self.max_int orelse 0)),
            .variance => self.computeVariance(false),
            .var_pop => self.computeVariance(true),
            .stddev => @sqrt(self.computeVariance(false)),
            .stddev_pop => @sqrt(self.computeVariance(true)),
            // Percentile-based aggregates use PercentileAccumulator, not this one
            .median, .percentile => unreachable,
        };
    }

    pub fn getIntResult(self: Accumulator) i64 {
        return switch (self.agg_type) {
            .count, .count_star => self.count,
            .sum => @as(i64, @intFromFloat(self.sum)),
            .avg => if (self.count > 0) @as(i64, @intFromFloat(self.sum / @as(f64, @floatFromInt(self.count)))) else 0,
            .min => self.min_int orelse 0,
            .max => self.max_int orelse 0,
            .variance, .var_pop, .stddev, .stddev_pop => @as(i64, @intFromFloat(self.getResult())),
            // Percentile-based aggregates use PercentileAccumulator, not this one
            .median, .percentile => unreachable,
        };
    }
};

/// Accumulator for percentile-based aggregates (MEDIAN, PERCENTILE)
/// These require storing all values to compute the result
pub const PercentileAccumulator = struct {
    allocator: std.mem.Allocator,
    values: std.ArrayList(f64),
    percentile: f64, // 0.5 for median, configurable for percentile

    pub fn init(allocator: std.mem.Allocator, pct: f64) PercentileAccumulator {
        return PercentileAccumulator{
            .allocator = allocator,
            .values = std.ArrayList(f64){},
            .percentile = pct,
        };
    }

    pub fn deinit(self: *PercentileAccumulator) void {
        self.values.deinit(self.allocator);
    }

    pub fn addValue(self: *PercentileAccumulator, value: f64) !void {
        try self.values.append(self.allocator, value);
    }

    pub fn addInt(self: *PercentileAccumulator, value: i64) !void {
        try self.addValue(@as(f64, @floatFromInt(value)));
    }

    pub fn addFloat(self: *PercentileAccumulator, value: f64) !void {
        try self.addValue(value);
    }

    /// Compute the percentile using linear interpolation
    pub fn getResult(self: *PercentileAccumulator) f64 {
        if (self.values.items.len == 0) return 0;

        // Sort values
        std.mem.sort(f64, self.values.items, {}, std.sort.asc(f64));

        const n = self.values.items.len;
        if (n == 1) return self.values.items[0];

        // Calculate position using linear interpolation
        const pos = self.percentile * @as(f64, @floatFromInt(n - 1));
        const lower_idx = @as(usize, @intFromFloat(@floor(pos)));
        const upper_idx = @min(lower_idx + 1, n - 1);
        const fraction = pos - @floor(pos);

        // Linear interpolation between lower and upper values
        const lower_val = self.values.items[lower_idx];
        const upper_val = self.values.items[upper_idx];
        return lower_val + fraction * (upper_val - lower_val);
    }
};

// ============================================================================
// Aggregate Detection Functions
// ============================================================================

/// Check if SELECT list contains any aggregate functions
pub fn hasAggregates(select_list: []const ast.SelectItem) bool {
    for (select_list) |item| {
        if (containsAggregate(&item.expr)) {
            return true;
        }
    }
    return false;
}

/// Recursively check if expression contains an aggregate function
pub fn containsAggregate(expr: *const Expr) bool {
    return switch (expr.*) {
        .call => |call| blk: {
            // Check if this is an aggregate function
            const is_agg = isAggregateFunction(call.name);
            if (is_agg) break :blk true;

            // Check arguments recursively
            for (call.args) |*arg| {
                if (containsAggregate(arg)) break :blk true;
            }
            break :blk false;
        },
        .binary => |bin| containsAggregate(bin.left) or containsAggregate(bin.right),
        .unary => |un| containsAggregate(un.operand),
        else => false,
    };
}

/// Check if function name is an aggregate function
pub fn isAggregateFunction(name: []const u8) bool {
    // Case-insensitive comparison
    if (name.len < 3 or name.len > 15) return false;

    var upper_buf: [16]u8 = undefined;
    const len = @min(name.len, upper_buf.len);
    const upper_name = std.ascii.upperString(upper_buf[0..len], name[0..len]);

    return std.mem.eql(u8, upper_name, "COUNT") or
        std.mem.eql(u8, upper_name, "SUM") or
        std.mem.eql(u8, upper_name, "AVG") or
        std.mem.eql(u8, upper_name, "MIN") or
        std.mem.eql(u8, upper_name, "MAX") or
        std.mem.eql(u8, upper_name, "STDDEV") or
        std.mem.eql(u8, upper_name, "STDDEV_SAMP") or
        std.mem.eql(u8, upper_name, "STDDEV_POP") or
        std.mem.eql(u8, upper_name, "VARIANCE") or
        std.mem.eql(u8, upper_name, "VAR_SAMP") or
        std.mem.eql(u8, upper_name, "VAR_POP") or
        std.mem.eql(u8, upper_name, "MEDIAN") or
        std.mem.eql(u8, upper_name, "PERCENTILE") or
        std.mem.eql(u8, upper_name, "PERCENTILE_CONT") or
        std.mem.eql(u8, upper_name, "QUANTILE");
}

/// Parse aggregate function name to AggregateType
pub fn parseAggregateType(name: []const u8) ?AggregateType {
    var upper_buf: [16]u8 = undefined;
    const len = @min(name.len, upper_buf.len);
    const upper_name = std.ascii.upperString(upper_buf[0..len], name[0..len]);

    if (std.mem.eql(u8, upper_name, "COUNT")) return .count;
    if (std.mem.eql(u8, upper_name, "SUM")) return .sum;
    if (std.mem.eql(u8, upper_name, "AVG")) return .avg;
    if (std.mem.eql(u8, upper_name, "MIN")) return .min;
    if (std.mem.eql(u8, upper_name, "MAX")) return .max;
    if (std.mem.eql(u8, upper_name, "STDDEV") or std.mem.eql(u8, upper_name, "STDDEV_SAMP")) return .stddev;
    if (std.mem.eql(u8, upper_name, "STDDEV_POP")) return .stddev_pop;
    if (std.mem.eql(u8, upper_name, "VARIANCE") or std.mem.eql(u8, upper_name, "VAR_SAMP")) return .variance;
    if (std.mem.eql(u8, upper_name, "VAR_POP")) return .var_pop;
    if (std.mem.eql(u8, upper_name, "MEDIAN")) return .median;
    if (std.mem.eql(u8, upper_name, "PERCENTILE") or std.mem.eql(u8, upper_name, "PERCENTILE_CONT") or std.mem.eql(u8, upper_name, "QUANTILE")) return .percentile;

    return null;
}

/// Parse aggregate function name to AggregateType with args (distinguishes COUNT(*) from COUNT(col))
pub fn parseAggregateTypeWithArgs(name: []const u8, args: []const Expr) AggregateType {
    var upper_buf: [16]u8 = undefined;
    const len = @min(name.len, upper_buf.len);
    const upper_name = std.ascii.upperString(upper_buf[0..len], name[0..len]);

    if (std.mem.eql(u8, upper_name, "COUNT")) {
        // COUNT(*) vs COUNT(col)
        if (args.len == 1 and args[0] == .column and
            std.mem.eql(u8, args[0].column.name, "*"))
        {
            return .count_star;
        }
        return .count;
    } else if (std.mem.eql(u8, upper_name, "SUM")) {
        return .sum;
    } else if (std.mem.eql(u8, upper_name, "AVG")) {
        return .avg;
    } else if (std.mem.eql(u8, upper_name, "MIN")) {
        return .min;
    } else if (std.mem.eql(u8, upper_name, "MAX")) {
        return .max;
    } else if (std.mem.eql(u8, upper_name, "STDDEV") or std.mem.eql(u8, upper_name, "STDDEV_SAMP")) {
        return .stddev;
    } else if (std.mem.eql(u8, upper_name, "STDDEV_POP")) {
        return .stddev_pop;
    } else if (std.mem.eql(u8, upper_name, "VARIANCE") or std.mem.eql(u8, upper_name, "VAR_SAMP")) {
        return .variance;
    } else if (std.mem.eql(u8, upper_name, "VAR_POP")) {
        return .var_pop;
    } else if (std.mem.eql(u8, upper_name, "MEDIAN")) {
        return .median;
    } else if (std.mem.eql(u8, upper_name, "PERCENTILE") or
        std.mem.eql(u8, upper_name, "PERCENTILE_CONT") or
        std.mem.eql(u8, upper_name, "QUANTILE"))
    {
        return .percentile;
    }
    return .count; // Default fallback
}

// ============================================================================
// Expression Matching Utilities
// ============================================================================

/// Check if two aggregate argument lists match
pub fn aggregateArgsMatch(a: []const Expr, b: []const Expr) bool {
    if (a.len != b.len) return false;

    for (a, b) |arg_a, arg_b| {
        if (!exprEquals(&arg_a, &arg_b)) return false;
    }

    return true;
}

/// Check if two expressions are equal (for aggregate matching)
pub fn exprEquals(a: *const Expr, b: *const Expr) bool {
    if (std.meta.activeTag(a.*) != std.meta.activeTag(b.*)) return false;

    return switch (a.*) {
        .column => |col_a| blk: {
            const col_b = b.column;
            break :blk std.mem.eql(u8, col_a.name, col_b.name);
        },
        .value => |val_a| blk: {
            const val_b = b.value;
            if (std.meta.activeTag(val_a) != std.meta.activeTag(val_b)) break :blk false;
            break :blk switch (val_a) {
                .integer => |i| i == val_b.integer,
                .float => |f| f == val_b.float,
                .string => |s| std.mem.eql(u8, s, val_b.string),
                .null => true,
                else => false,
            };
        },
        else => false,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "aggregate: Accumulator SUM" {
    var acc = Accumulator.init(.sum);
    acc.addInt(1);
    acc.addInt(2);
    acc.addInt(3);
    try std.testing.expectEqual(@as(f64, 6.0), acc.getResult());

    // Test with floats
    var acc_f = Accumulator.init(.sum);
    acc_f.addFloat(1.5);
    acc_f.addFloat(2.5);
    acc_f.addFloat(3.0);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), acc_f.getResult(), 0.001);
}

test "aggregate: Accumulator AVG" {
    var acc = Accumulator.init(.avg);
    acc.addFloat(1.0);
    acc.addFloat(2.0);
    acc.addFloat(3.0);
    try std.testing.expectEqual(@as(f64, 2.0), acc.getResult());

    // Empty accumulator
    var empty = Accumulator.init(.avg);
    try std.testing.expectEqual(@as(f64, 0.0), empty.getResult());
}

test "aggregate: Accumulator MIN/MAX" {
    var acc_min = Accumulator.init(.min);
    var acc_max = Accumulator.init(.max);

    acc_min.addInt(5);
    acc_min.addInt(2);
    acc_min.addInt(8);
    acc_min.addInt(1);

    acc_max.addInt(5);
    acc_max.addInt(2);
    acc_max.addInt(8);
    acc_max.addInt(1);

    try std.testing.expectEqual(@as(i64, 1), acc_min.getIntResult());
    try std.testing.expectEqual(@as(i64, 8), acc_max.getIntResult());

    // Test with floats
    var acc_min_f = Accumulator.init(.min);
    acc_min_f.addFloat(5.5);
    acc_min_f.addFloat(2.2);
    acc_min_f.addFloat(8.8);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), acc_min_f.getResult(), 0.001);
}

test "aggregate: Accumulator COUNT" {
    var acc = Accumulator.init(.count);
    acc.addCount();
    acc.addCount();
    acc.addCount();
    try std.testing.expectEqual(@as(i64, 3), acc.count);
    try std.testing.expectEqual(@as(f64, 3.0), acc.getResult());

    var acc_star = Accumulator.init(.count_star);
    acc_star.addCount();
    acc_star.addCount();
    try std.testing.expectEqual(@as(i64, 2), acc_star.count);
}

test "aggregate: Accumulator STDDEV and VARIANCE" {
    // Values: 2, 4, 4, 4, 5, 5, 7, 9
    // Mean = 5, Sample variance = 4.571, Sample stddev = 2.138
    var acc_var = Accumulator.init(.variance);
    var acc_std = Accumulator.init(.stddev);

    const values = [_]f64{ 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };
    for (values) |v| {
        acc_var.addFloat(v);
        acc_std.addFloat(v);
    }

    try std.testing.expectApproxEqAbs(@as(f64, 4.571), acc_var.getResult(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.138), acc_std.getResult(), 0.01);

    // Population variance/stddev
    var acc_var_pop = Accumulator.init(.var_pop);
    var acc_std_pop = Accumulator.init(.stddev_pop);
    for (values) |v| {
        acc_var_pop.addFloat(v);
        acc_std_pop.addFloat(v);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), acc_var_pop.getResult(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), acc_std_pop.getResult(), 0.01);
}

test "aggregate: PercentileAccumulator median" {
    const allocator = std.testing.allocator;

    // Odd number of values - median is middle value
    var acc = PercentileAccumulator.init(allocator, 0.5);
    defer acc.deinit();
    try acc.addFloat(1.0);
    try acc.addFloat(5.0);
    try acc.addFloat(3.0);
    try std.testing.expectEqual(@as(f64, 3.0), acc.getResult());

    // Even number of values - median is average of two middle values
    var acc2 = PercentileAccumulator.init(allocator, 0.5);
    defer acc2.deinit();
    try acc2.addFloat(1.0);
    try acc2.addFloat(2.0);
    try acc2.addFloat(3.0);
    try acc2.addFloat(4.0);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), acc2.getResult(), 0.001);
}

test "aggregate: isAggregateFunction" {
    // Positive cases
    try std.testing.expect(isAggregateFunction("COUNT"));
    try std.testing.expect(isAggregateFunction("count"));
    try std.testing.expect(isAggregateFunction("SUM"));
    try std.testing.expect(isAggregateFunction("sum"));
    try std.testing.expect(isAggregateFunction("AVG"));
    try std.testing.expect(isAggregateFunction("MIN"));
    try std.testing.expect(isAggregateFunction("MAX"));
    try std.testing.expect(isAggregateFunction("STDDEV"));
    try std.testing.expect(isAggregateFunction("VARIANCE"));
    try std.testing.expect(isAggregateFunction("MEDIAN"));

    // Negative cases
    try std.testing.expect(!isAggregateFunction("UPPER"));
    try std.testing.expect(!isAggregateFunction("LOWER"));
    try std.testing.expect(!isAggregateFunction("LENGTH"));
    try std.testing.expect(!isAggregateFunction("TRIM"));
    try std.testing.expect(!isAggregateFunction("AB")); // Too short
}

test "aggregate: parseAggregateType" {
    try std.testing.expectEqual(AggregateType.count, parseAggregateType("COUNT").?);
    try std.testing.expectEqual(AggregateType.sum, parseAggregateType("SUM").?);
    try std.testing.expectEqual(AggregateType.sum, parseAggregateType("sum").?);
    try std.testing.expectEqual(AggregateType.avg, parseAggregateType("AVG").?);
    try std.testing.expectEqual(AggregateType.min, parseAggregateType("MIN").?);
    try std.testing.expectEqual(AggregateType.max, parseAggregateType("MAX").?);
    try std.testing.expectEqual(AggregateType.stddev, parseAggregateType("STDDEV").?);
    try std.testing.expectEqual(AggregateType.stddev, parseAggregateType("STDDEV_SAMP").?);
    try std.testing.expectEqual(AggregateType.stddev_pop, parseAggregateType("STDDEV_POP").?);
    try std.testing.expectEqual(AggregateType.variance, parseAggregateType("VARIANCE").?);
    try std.testing.expectEqual(AggregateType.median, parseAggregateType("MEDIAN").?);
    try std.testing.expectEqual(AggregateType.percentile, parseAggregateType("PERCENTILE").?);
    try std.testing.expectEqual(@as(?AggregateType, null), parseAggregateType("UPPER"));
    try std.testing.expectEqual(@as(?AggregateType, null), parseAggregateType("INVALID"));
}
