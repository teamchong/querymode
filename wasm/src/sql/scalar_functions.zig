//! Scalar Functions - Date/Time, String, and Math functions for SQL expressions
//!
//! These are standalone functions that can be called from the SQL executor.
//! Functions that need memory allocation take an allocator parameter.

const std = @import("std");
const ast = @import("ast");
const Value = ast.Value;

// ============================================================================
// Date/Time Functions
// ============================================================================

/// Date part enum for date/time functions
pub const DatePart = enum {
    year,
    month,
    day,
    hour,
    minute,
    second,
    millisecond,
    dayofweek,
    dayofyear,
    week,
    quarter,

    /// Map shorthand function names to DatePart (e.g., "YEAR" -> .year)
    pub fn fromFunctionName(upper_name: []const u8) ?DatePart {
        if (std.mem.eql(u8, upper_name, "YEAR")) return .year;
        if (std.mem.eql(u8, upper_name, "MONTH")) return .month;
        if (std.mem.eql(u8, upper_name, "DAY")) return .day;
        if (std.mem.eql(u8, upper_name, "HOUR")) return .hour;
        if (std.mem.eql(u8, upper_name, "MINUTE")) return .minute;
        if (std.mem.eql(u8, upper_name, "SECOND")) return .second;
        if (std.mem.eql(u8, upper_name, "DAYOFWEEK") or std.mem.eql(u8, upper_name, "DOW")) return .dayofweek;
        if (std.mem.eql(u8, upper_name, "DAYOFYEAR") or std.mem.eql(u8, upper_name, "DOY")) return .dayofyear;
        if (std.mem.eql(u8, upper_name, "WEEK")) return .week;
        if (std.mem.eql(u8, upper_name, "QUARTER")) return .quarter;
        return null;
    }
};

/// Date components for conversions
pub const DateComponents = struct {
    year: i64,
    month: i64,
    day: i64,
    dayOfYear: i64,
};

/// Parse date part from string value
pub fn parseDatePart(val: Value) ?DatePart {
    const part_str = switch (val) {
        .string => |s| s,
        else => return null,
    };

    var upper_buf: [16]u8 = undefined;
    const len = @min(part_str.len, upper_buf.len);
    const upper = std.ascii.upperString(upper_buf[0..len], part_str[0..len]);

    if (std.mem.eql(u8, upper, "YEAR") or std.mem.eql(u8, upper, "Y")) return .year;
    if (std.mem.eql(u8, upper, "MONTH") or std.mem.eql(u8, upper, "M")) return .month;
    if (std.mem.eql(u8, upper, "DAY") or std.mem.eql(u8, upper, "D")) return .day;
    if (std.mem.eql(u8, upper, "HOUR") or std.mem.eql(u8, upper, "H")) return .hour;
    if (std.mem.eql(u8, upper, "MINUTE") or std.mem.eql(u8, upper, "MI")) return .minute;
    if (std.mem.eql(u8, upper, "SECOND") or std.mem.eql(u8, upper, "S")) return .second;
    if (std.mem.eql(u8, upper, "MILLISECOND") or std.mem.eql(u8, upper, "MS")) return .millisecond;
    if (std.mem.eql(u8, upper, "DAYOFWEEK") or std.mem.eql(u8, upper, "DOW")) return .dayofweek;
    if (std.mem.eql(u8, upper, "DAYOFYEAR") or std.mem.eql(u8, upper, "DOY")) return .dayofyear;
    if (std.mem.eql(u8, upper, "WEEK") or std.mem.eql(u8, upper, "W")) return .week;
    if (std.mem.eql(u8, upper, "QUARTER") or std.mem.eql(u8, upper, "Q")) return .quarter;

    return null;
}

/// Convert timestamp to seconds since epoch
pub fn toEpochSeconds(val: Value) ?i64 {
    return switch (val) {
        .integer => |i| i, // Assume already in seconds
        .float => |f| @intFromFloat(f),
        else => null,
    };
}

/// Convert days since epoch to year/month/day
/// Algorithm based on Howard Hinnant's date algorithms
pub fn daysToDate(days_since_epoch: i64) DateComponents {
    const z = days_since_epoch + 719468; // Days since Mar 1, 0000
    const era: i64 = if (z >= 0) @divFloor(z, 146097) else @divFloor(z - 146096, 146097);
    const doe: i64 = z - era * 146097; // Day of era [0, 146096]
    const yoe: i64 = @divFloor(doe - @divFloor(doe, 1460) + @divFloor(doe, 36524) - @divFloor(doe, 146096), 365);
    const y: i64 = yoe + era * 400;
    const doy: i64 = doe - (365 * yoe + @divFloor(yoe, 4) - @divFloor(yoe, 100)); // Day of year [0, 365]
    const mp: i64 = @divFloor(5 * doy + 2, 153);
    const d: i64 = doy - @divFloor(153 * mp + 2, 5) + 1;
    const m: i64 = if (mp < 10) mp + 3 else mp - 9;
    const year = if (m <= 2) y + 1 else y;

    // Calculate day of year for the actual year
    const jan1_days = dateToDays(year, 1, 1);
    const day_of_year = days_since_epoch - jan1_days + 1;

    return .{
        .year = year,
        .month = m,
        .day = d,
        .dayOfYear = day_of_year,
    };
}

/// Convert year/month/day to days since epoch
pub fn dateToDays(year: i64, month: i64, day: i64) i64 {
    const y = if (month <= 2) year - 1 else year;
    const era: i64 = if (y >= 0) @divFloor(y, 400) else @divFloor(y - 399, 400);
    const yoe: i64 = y - era * 400;
    const m = if (month <= 2) month + 9 else month - 3;
    const doy: i64 = @divFloor(153 * m + 2, 5) + day - 1;
    const doe: i64 = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

/// EXTRACT(part, timestamp) - Extract date/time component
pub fn funcExtract(part_val: Value, ts_val: Value) Value {
    const part = parseDatePart(part_val) orelse return Value{ .null = {} };
    return funcExtractPart(part, ts_val);
}

/// Extract a specific date part from timestamp
pub fn funcExtractPart(part: DatePart, ts_val: Value) Value {
    const epoch_secs = toEpochSeconds(ts_val) orelse return Value{ .null = {} };

    // Convert epoch seconds to date components
    const secs_per_day: i64 = 86400;
    const secs_per_hour: i64 = 3600;
    const secs_per_min: i64 = 60;

    // Days since epoch (Jan 1, 1970)
    var days = @divFloor(epoch_secs, secs_per_day);
    var remaining = @mod(epoch_secs, secs_per_day);
    if (remaining < 0) {
        remaining += secs_per_day;
        days -= 1;
    }

    const hour = @divFloor(remaining, secs_per_hour);
    remaining = @mod(remaining, secs_per_hour);
    const minute = @divFloor(remaining, secs_per_min);
    const second = @mod(remaining, secs_per_min);

    // Calculate year, month, day from days since epoch
    const date = daysToDate(days);

    return switch (part) {
        .year => Value{ .integer = date.year },
        .month => Value{ .integer = date.month },
        .day => Value{ .integer = date.day },
        .hour => Value{ .integer = hour },
        .minute => Value{ .integer = minute },
        .second => Value{ .integer = second },
        .millisecond => Value{ .integer = 0 }, // Would need ms precision input
        .dayofweek => Value{ .integer = @mod(days + 4, 7) }, // Jan 1, 1970 was Thursday (4)
        .dayofyear => Value{ .integer = date.dayOfYear },
        .week => Value{ .integer = @divFloor(date.dayOfYear - 1, 7) + 1 },
        .quarter => Value{ .integer = @divFloor(date.month - 1, 3) + 1 },
    };
}

/// DATE_TRUNC(part, timestamp) - Truncate timestamp to precision
pub fn funcDateTrunc(part_val: Value, ts_val: Value) Value {
    const part = parseDatePart(part_val) orelse return Value{ .null = {} };
    const epoch_secs = toEpochSeconds(ts_val) orelse return Value{ .null = {} };

    const secs_per_day: i64 = 86400;
    const secs_per_hour: i64 = 3600;
    const secs_per_min: i64 = 60;

    const days = @divFloor(epoch_secs, secs_per_day);
    const date = daysToDate(days);

    return switch (part) {
        .year => Value{ .integer = dateToDays(date.year, 1, 1) * secs_per_day },
        .month => Value{ .integer = dateToDays(date.year, date.month, 1) * secs_per_day },
        .day => Value{ .integer = days * secs_per_day },
        .hour => Value{ .integer = @divFloor(epoch_secs, secs_per_hour) * secs_per_hour },
        .minute => Value{ .integer = @divFloor(epoch_secs, secs_per_min) * secs_per_min },
        .second => Value{ .integer = epoch_secs },
        else => Value{ .null = {} },
    };
}

/// DATE_ADD(timestamp, interval, part) - Add interval to timestamp
pub fn funcDateAdd(ts_val: Value, interval_val: Value, part_val: Value) Value {
    const part = parseDatePart(part_val) orelse return Value{ .null = {} };
    const epoch_secs = toEpochSeconds(ts_val) orelse return Value{ .null = {} };
    const interval = switch (interval_val) {
        .integer => |i| i,
        .float => |f| @as(i64, @intFromFloat(f)),
        else => return Value{ .null = {} },
    };

    const secs_per_day: i64 = 86400;
    const secs_per_hour: i64 = 3600;
    const secs_per_min: i64 = 60;

    return switch (part) {
        .year => blk: {
            const days = @divFloor(epoch_secs, secs_per_day);
            const time_of_day = @mod(epoch_secs, secs_per_day);
            const date = daysToDate(days);
            const new_days = dateToDays(date.year + interval, date.month, date.day);
            break :blk Value{ .integer = new_days * secs_per_day + time_of_day };
        },
        .month => blk: {
            const days = @divFloor(epoch_secs, secs_per_day);
            const time_of_day = @mod(epoch_secs, secs_per_day);
            const date = daysToDate(days);
            var new_month = date.month + interval;
            var new_year = date.year;
            while (new_month > 12) {
                new_month -= 12;
                new_year += 1;
            }
            while (new_month < 1) {
                new_month += 12;
                new_year -= 1;
            }
            const new_days = dateToDays(new_year, new_month, date.day);
            break :blk Value{ .integer = new_days * secs_per_day + time_of_day };
        },
        .day => Value{ .integer = epoch_secs + interval * secs_per_day },
        .hour => Value{ .integer = epoch_secs + interval * secs_per_hour },
        .minute => Value{ .integer = epoch_secs + interval * secs_per_min },
        .second => Value{ .integer = epoch_secs + interval },
        else => Value{ .null = {} },
    };
}

/// DATE_DIFF(timestamp1, timestamp2, part) - Difference in specified units
pub fn funcDateDiff(ts1_val: Value, ts2_val: Value, part_val: Value) Value {
    const part = parseDatePart(part_val) orelse return Value{ .null = {} };
    const epoch1 = toEpochSeconds(ts1_val) orelse return Value{ .null = {} };
    const epoch2 = toEpochSeconds(ts2_val) orelse return Value{ .null = {} };

    const diff = epoch1 - epoch2;
    const secs_per_day: i64 = 86400;
    const secs_per_hour: i64 = 3600;
    const secs_per_min: i64 = 60;

    return switch (part) {
        .year => blk: {
            const date1 = daysToDate(@divFloor(epoch1, secs_per_day));
            const date2 = daysToDate(@divFloor(epoch2, secs_per_day));
            break :blk Value{ .integer = date1.year - date2.year };
        },
        .month => blk: {
            const date1 = daysToDate(@divFloor(epoch1, secs_per_day));
            const date2 = daysToDate(@divFloor(epoch2, secs_per_day));
            break :blk Value{ .integer = (date1.year - date2.year) * 12 + (date1.month - date2.month) };
        },
        .day => Value{ .integer = @divFloor(diff, secs_per_day) },
        .hour => Value{ .integer = @divFloor(diff, secs_per_hour) },
        .minute => Value{ .integer = @divFloor(diff, secs_per_min) },
        .second => Value{ .integer = diff },
        else => Value{ .null = {} },
    };
}

/// EPOCH(timestamp) - Return epoch seconds
pub fn funcEpoch(val: Value) Value {
    return switch (val) {
        .integer => val,
        .float => |f| Value{ .integer = @intFromFloat(f) },
        else => Value{ .null = {} },
    };
}

// ============================================================================
// String Functions
// ============================================================================

/// UPPER(string) - Convert string to uppercase
pub fn funcUpper(allocator: std.mem.Allocator, val: Value) Value {
    const str = switch (val) {
        .string => |s| s,
        else => return Value{ .null = {} },
    };

    const result = allocator.alloc(u8, str.len) catch return Value{ .null = {} };
    for (str, 0..) |c, i| {
        result[i] = std.ascii.toUpper(c);
    }

    return Value{ .string = result };
}

/// LOWER(string) - Convert string to lowercase
pub fn funcLower(allocator: std.mem.Allocator, val: Value) Value {
    const str = switch (val) {
        .string => |s| s,
        else => return Value{ .null = {} },
    };

    const result = allocator.alloc(u8, str.len) catch return Value{ .null = {} };
    for (str, 0..) |c, i| {
        result[i] = std.ascii.toLower(c);
    }

    return Value{ .string = result };
}

/// LENGTH(string) - Return string length
pub fn funcLength(val: Value) Value {
    return switch (val) {
        .string => |s| Value{ .integer = @intCast(s.len) },
        else => Value{ .null = {} },
    };
}

/// TRIM(string) - Remove leading/trailing whitespace
pub fn funcTrim(allocator: std.mem.Allocator, val: Value) Value {
    const str = switch (val) {
        .string => |s| s,
        else => return Value{ .null = {} },
    };

    const trimmed = std.mem.trim(u8, str, " \t\n\r");
    const result = allocator.dupe(u8, trimmed) catch return Value{ .null = {} };

    return Value{ .string = result };
}

// ============================================================================
// Math Functions
// ============================================================================

/// ABS(number) - Absolute value
pub fn funcAbs(val: Value) Value {
    return switch (val) {
        .integer => |i| Value{ .integer = if (i < 0) -i else i },
        .float => |f| Value{ .float = @abs(f) },
        else => Value{ .null = {} },
    };
}

/// ROUND(number, precision) - Round to precision decimal places
pub fn funcRound(val: Value, precision: i32) Value {
    const f = switch (val) {
        .integer => |i| @as(f64, @floatFromInt(i)),
        .float => |fl| fl,
        else => return Value{ .null = {} },
    };

    const multiplier = std.math.pow(f64, 10.0, @floatFromInt(precision));
    return Value{ .float = @round(f * multiplier) / multiplier };
}

/// FLOOR(number) - Round down
pub fn funcFloor(val: Value) Value {
    return switch (val) {
        .integer => val,
        .float => |f| Value{ .float = @floor(f) },
        else => Value{ .null = {} },
    };
}

/// CEIL(number) - Round up
pub fn funcCeil(val: Value) Value {
    return switch (val) {
        .integer => val,
        .float => |f| Value{ .float = @ceil(f) },
        else => Value{ .null = {} },
    };
}

// ============================================================================
// Type Inference
// ============================================================================

/// Result type enum for expression type inference
pub const ResultType = enum {
    int64,
    float64,
    string,
};

/// Infer return type of a scalar function by name
pub fn inferFunctionReturnType(name: []const u8) ResultType {
    var upper_buf: [32]u8 = undefined;
    const upper_name = std.ascii.upperString(&upper_buf, name);

    // String functions return string
    if (std.mem.eql(u8, upper_name, "UPPER") or
        std.mem.eql(u8, upper_name, "LOWER") or
        std.mem.eql(u8, upper_name, "TRIM"))
    {
        return .string;
    }

    // LENGTH returns int
    if (std.mem.eql(u8, upper_name, "LENGTH")) {
        return .int64;
    }

    // Date/Time functions return int64
    if (std.mem.eql(u8, upper_name, "YEAR") or
        std.mem.eql(u8, upper_name, "MONTH") or
        std.mem.eql(u8, upper_name, "DAY") or
        std.mem.eql(u8, upper_name, "HOUR") or
        std.mem.eql(u8, upper_name, "MINUTE") or
        std.mem.eql(u8, upper_name, "SECOND") or
        std.mem.eql(u8, upper_name, "DAYOFWEEK") or
        std.mem.eql(u8, upper_name, "DOW") or
        std.mem.eql(u8, upper_name, "DAYOFYEAR") or
        std.mem.eql(u8, upper_name, "DOY") or
        std.mem.eql(u8, upper_name, "WEEK") or
        std.mem.eql(u8, upper_name, "QUARTER") or
        std.mem.eql(u8, upper_name, "EXTRACT") or
        std.mem.eql(u8, upper_name, "DATE_PART") or
        std.mem.eql(u8, upper_name, "DATE_TRUNC") or
        std.mem.eql(u8, upper_name, "DATE_ADD") or
        std.mem.eql(u8, upper_name, "DATEADD") or
        std.mem.eql(u8, upper_name, "DATE_DIFF") or
        std.mem.eql(u8, upper_name, "DATEDIFF") or
        std.mem.eql(u8, upper_name, "EPOCH") or
        std.mem.eql(u8, upper_name, "UNIX_TIMESTAMP") or
        std.mem.eql(u8, upper_name, "FROM_UNIXTIME") or
        std.mem.eql(u8, upper_name, "TO_TIMESTAMP"))
    {
        return .int64;
    }

    // Math functions typically return float (except ABS which preserves type)
    // For simplicity, return float64 for all math functions
    return .float64;
}

// ============================================================================
// Comparison Functions
// ============================================================================

const BinaryOp = @import("ast").BinaryOp;

/// Compare two numbers
pub fn compareNumbers(op: BinaryOp, left: f64, right: f64) bool {
    return switch (op) {
        .eq => left == right,
        .ne => left != right,
        .lt => left < right,
        .le => left <= right,
        .gt => left > right,
        .ge => left >= right,
        else => unreachable,
    };
}

/// Compare two strings
pub fn compareStrings(op: BinaryOp, left: []const u8, right: []const u8) bool {
    const cmp = std.mem.order(u8, left, right);
    return switch (op) {
        .eq => cmp == .eq,
        .ne => cmp != .eq,
        .lt => cmp == .lt,
        .le => cmp == .lt or cmp == .eq,
        .gt => cmp == .gt,
        .ge => cmp == .gt or cmp == .eq,
        else => unreachable,
    };
}

// ============================================================================
// Value Arithmetic Functions
// ============================================================================

/// Negate a numeric value
pub fn negateValue(val: Value) Value {
    return switch (val) {
        .integer => |i| Value{ .integer = -i },
        .float => |f| Value{ .float = -f },
        else => Value{ .null = {} },
    };
}

/// Add two values (int + int = int, int + float = float, float + float = float)
pub fn addValues(left: Value, right: Value) Value {
    return switch (left) {
        .integer => |l| switch (right) {
            .integer => |r| Value{ .integer = l + r },
            .float => |r| Value{ .float = @as(f64, @floatFromInt(l)) + r },
            else => Value{ .null = {} },
        },
        .float => |l| switch (right) {
            .integer => |r| Value{ .float = l + @as(f64, @floatFromInt(r)) },
            .float => |r| Value{ .float = l + r },
            else => Value{ .null = {} },
        },
        else => Value{ .null = {} },
    };
}

/// Subtract two values
pub fn subtractValues(left: Value, right: Value) Value {
    return switch (left) {
        .integer => |l| switch (right) {
            .integer => |r| Value{ .integer = l - r },
            .float => |r| Value{ .float = @as(f64, @floatFromInt(l)) - r },
            else => Value{ .null = {} },
        },
        .float => |l| switch (right) {
            .integer => |r| Value{ .float = l - @as(f64, @floatFromInt(r)) },
            .float => |r| Value{ .float = l - r },
            else => Value{ .null = {} },
        },
        else => Value{ .null = {} },
    };
}

/// Multiply two values
pub fn multiplyValues(left: Value, right: Value) Value {
    return switch (left) {
        .integer => |l| switch (right) {
            .integer => |r| Value{ .integer = l * r },
            .float => |r| Value{ .float = @as(f64, @floatFromInt(l)) * r },
            else => Value{ .null = {} },
        },
        .float => |l| switch (right) {
            .integer => |r| Value{ .float = l * @as(f64, @floatFromInt(r)) },
            .float => |r| Value{ .float = l * r },
            else => Value{ .null = {} },
        },
        else => Value{ .null = {} },
    };
}

/// Divide two values (always returns float for precision)
pub fn divideValues(left: Value, right: Value) Value {
    const left_f = switch (left) {
        .integer => |i| @as(f64, @floatFromInt(i)),
        .float => |f| f,
        else => return Value{ .null = {} },
    };
    const right_f = switch (right) {
        .integer => |i| @as(f64, @floatFromInt(i)),
        .float => |f| f,
        else => return Value{ .null = {} },
    };

    if (right_f == 0) return Value{ .null = {} }; // Division by zero
    return Value{ .float = left_f / right_f };
}

/// Modulo two values
pub fn moduloValues(left: Value, right: Value) Value {
    return switch (left) {
        .integer => |l| switch (right) {
            .integer => |r| if (r != 0) Value{ .integer = @mod(l, r) } else Value{ .null = {} },
            else => Value{ .null = {} },
        },
        .float => |l| switch (right) {
            .float => |r| if (r != 0) Value{ .float = @mod(l, r) } else Value{ .null = {} },
            .integer => |r| if (r != 0) Value{ .float = @mod(l, @as(f64, @floatFromInt(r))) } else Value{ .null = {} },
            else => Value{ .null = {} },
        },
        else => Value{ .null = {} },
    };
}

// ============================================================================
// Tests
// ============================================================================

test "scalar: parseDatePart" {
    try std.testing.expectEqual(DatePart.year, parseDatePart(Value{ .string = "YEAR" }).?);
    try std.testing.expectEqual(DatePart.month, parseDatePart(Value{ .string = "MONTH" }).?);
    try std.testing.expectEqual(DatePart.day, parseDatePart(Value{ .string = "DAY" }).?);
    try std.testing.expectEqual(DatePart.hour, parseDatePart(Value{ .string = "HOUR" }).?);
    try std.testing.expectEqual(DatePart.minute, parseDatePart(Value{ .string = "MINUTE" }).?);
    try std.testing.expectEqual(DatePart.second, parseDatePart(Value{ .string = "SECOND" }).?);
    try std.testing.expectEqual(@as(?DatePart, null), parseDatePart(Value{ .string = "INVALID" }));
    try std.testing.expectEqual(@as(?DatePart, null), parseDatePart(Value{ .integer = 42 }));
}

test "scalar: daysToDate and dateToDays roundtrip" {
    // Known date: 2024-01-15 = 19737 days since epoch (1970-01-01)
    const days: i64 = 19737;
    const date = daysToDate(days);
    try std.testing.expectEqual(@as(i64, 2024), date.year);
    try std.testing.expectEqual(@as(i64, 1), date.month);
    try std.testing.expectEqual(@as(i64, 15), date.day);

    // Roundtrip
    const back = dateToDays(date.year, date.month, date.day);
    try std.testing.expectEqual(days, back);

    // Test epoch (1970-01-01)
    const epoch = daysToDate(0);
    try std.testing.expectEqual(@as(i64, 1970), epoch.year);
    try std.testing.expectEqual(@as(i64, 1), epoch.month);
    try std.testing.expectEqual(@as(i64, 1), epoch.day);
}

test "scalar: funcUpper" {
    const allocator = std.testing.allocator;
    const result = funcUpper(allocator, Value{ .string = "hello" });
    defer if (result == .string) allocator.free(result.string);
    try std.testing.expectEqualStrings("HELLO", result.string);

    // Test null handling
    try std.testing.expect(funcUpper(allocator, Value{ .null = {} }) == .null);
    try std.testing.expect(funcUpper(allocator, Value{ .integer = 42 }) == .null);
}

test "scalar: funcLower" {
    const allocator = std.testing.allocator;
    const result = funcLower(allocator, Value{ .string = "HELLO" });
    defer if (result == .string) allocator.free(result.string);
    try std.testing.expectEqualStrings("hello", result.string);

    // Test mixed case
    const mixed = funcLower(allocator, Value{ .string = "HeLLo WoRLd" });
    defer if (mixed == .string) allocator.free(mixed.string);
    try std.testing.expectEqualStrings("hello world", mixed.string);
}

test "scalar: funcLength" {
    try std.testing.expectEqual(@as(i64, 5), funcLength(Value{ .string = "hello" }).integer);
    try std.testing.expectEqual(@as(i64, 0), funcLength(Value{ .string = "" }).integer);
    try std.testing.expectEqual(@as(i64, 11), funcLength(Value{ .string = "hello world" }).integer);
    try std.testing.expect(funcLength(Value{ .integer = 42 }) == .null);
    try std.testing.expect(funcLength(Value{ .null = {} }) == .null);
}

test "scalar: funcTrim" {
    const allocator = std.testing.allocator;

    const result1 = funcTrim(allocator, Value{ .string = "  hello  " });
    defer if (result1 == .string) allocator.free(result1.string);
    try std.testing.expectEqualStrings("hello", result1.string);

    const result2 = funcTrim(allocator, Value{ .string = "\t\nhello\r\n" });
    defer if (result2 == .string) allocator.free(result2.string);
    try std.testing.expectEqualStrings("hello", result2.string);

    try std.testing.expect(funcTrim(allocator, Value{ .null = {} }) == .null);
}

test "scalar: funcAbs" {
    try std.testing.expectEqual(@as(i64, 5), funcAbs(Value{ .integer = -5 }).integer);
    try std.testing.expectEqual(@as(i64, 5), funcAbs(Value{ .integer = 5 }).integer);
    try std.testing.expectEqual(@as(i64, 0), funcAbs(Value{ .integer = 0 }).integer);
    try std.testing.expectApproxEqAbs(@as(f64, 3.14), funcAbs(Value{ .float = -3.14 }).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 3.14), funcAbs(Value{ .float = 3.14 }).float, 0.001);
    try std.testing.expect(funcAbs(Value{ .null = {} }) == .null);
}

test "scalar: funcRound" {
    try std.testing.expectApproxEqAbs(@as(f64, 3.14), funcRound(Value{ .float = 3.14159 }, 2).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), funcRound(Value{ .float = 3.14159 }, 0).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 3.1), funcRound(Value{ .float = 3.14159 }, 1).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), funcRound(Value{ .integer = 5 }, 2).float, 0.001);
    try std.testing.expect(funcRound(Value{ .null = {} }, 2) == .null);
}

test "scalar: funcFloor and funcCeil" {
    // Floor tests
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), funcFloor(Value{ .float = 3.7 }).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), funcFloor(Value{ .float = 3.2 }).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, -4.0), funcFloor(Value{ .float = -3.2 }).float, 0.001);
    try std.testing.expectEqual(@as(i64, 5), funcFloor(Value{ .integer = 5 }).integer);

    // Ceil tests
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), funcCeil(Value{ .float = 3.2 }).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), funcCeil(Value{ .float = 3.7 }).float, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, -3.0), funcCeil(Value{ .float = -3.2 }).float, 0.001);
    try std.testing.expectEqual(@as(i64, 5), funcCeil(Value{ .integer = 5 }).integer);
}

test "scalar: arithmetic operations" {
    // Addition
    try std.testing.expectEqual(@as(i64, 5), addValues(Value{ .integer = 2 }, Value{ .integer = 3 }).integer);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), addValues(Value{ .float = 2.5 }, Value{ .float = 3.0 }).float, 0.001);

    // Subtraction
    try std.testing.expectEqual(@as(i64, 2), subtractValues(Value{ .integer = 5 }, Value{ .integer = 3 }).integer);

    // Multiplication
    try std.testing.expectEqual(@as(i64, 6), multiplyValues(Value{ .integer = 2 }, Value{ .integer = 3 }).integer);

    // Division
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), divideValues(Value{ .integer = 5 }, Value{ .integer = 2 }).float, 0.001);
    try std.testing.expect(divideValues(Value{ .integer = 5 }, Value{ .integer = 0 }) == .null);

    // Modulo
    try std.testing.expectEqual(@as(i64, 1), moduloValues(Value{ .integer = 5 }, Value{ .integer = 2 }).integer);
}

test "scalar: null handling" {
    try std.testing.expect(funcLength(Value{ .null = {} }) == .null);
    try std.testing.expect(funcAbs(Value{ .null = {} }) == .null);
    try std.testing.expect(funcFloor(Value{ .null = {} }) == .null);
    try std.testing.expect(funcCeil(Value{ .null = {} }) == .null);
    try std.testing.expect(addValues(Value{ .null = {} }, Value{ .integer = 5 }) == .null);
    try std.testing.expect(addValues(Value{ .integer = 5 }, Value{ .null = {} }) == .null);
}
