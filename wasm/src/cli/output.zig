//! CLI Output Formatting
//!
//! Provides result formatting for table, CSV, and JSON output.

const std = @import("std");
const executor = @import("lanceql.sql.executor");

/// Output query results in the specified format
pub fn outputResults(result: *executor.Result, json: bool, csv: bool) void {
    if (json) {
        printResultsJson(result);
    } else if (csv) {
        printResultsCsv(result);
    } else {
        printResultsTable(result);
    }
}

fn printResultsDelimited(result: *executor.Result, comptime delimiter: []const u8) void {
    // Print header
    for (result.columns, 0..) |col, i| {
        if (i > 0) std.debug.print(delimiter, .{});
        std.debug.print("{s}", .{col.name});
    }
    std.debug.print("\n", .{});

    // Print rows
    for (0..result.row_count) |row| {
        for (result.columns, 0..) |col, i| {
            if (i > 0) std.debug.print(delimiter, .{});
            printValue(col.data, row);
        }
        std.debug.print("\n", .{});
    }
}

pub fn printResultsTable(result: *executor.Result) void {
    printResultsDelimited(result, "\t");
}

pub fn printResultsCsv(result: *executor.Result) void {
    printResultsDelimited(result, ",");
}

pub fn printResultsJson(result: *executor.Result) void {
    std.debug.print("[", .{});
    for (0..result.row_count) |row| {
        if (row > 0) std.debug.print(",", .{});
        std.debug.print("{{", .{});
        for (result.columns, 0..) |col, i| {
            if (i > 0) std.debug.print(",", .{});
            std.debug.print("\"{s}\":", .{col.name});
            printValueJson(col.data, row);
        }
        std.debug.print("}}", .{});
    }
    std.debug.print("]\n", .{});
}

fn printValue(data: executor.Result.ColumnData, row: usize) void {
    switch (data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .int32, .date32 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .float64 => |arr| {
            std.debug.print("{d:.6}", .{arr[row]});
        },
        .float32 => |arr| {
            std.debug.print("{d:.6}", .{arr[row]});
        },
        .bool_ => |arr| {
            std.debug.print("{}", .{arr[row]});
        },
        .string => |arr| {
            std.debug.print("{s}", .{arr[row]});
        },
    }
}

fn printValueJson(data: executor.Result.ColumnData, row: usize) void {
    switch (data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .int32, .date32 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .float64 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .float32 => |arr| {
            std.debug.print("{d}", .{arr[row]});
        },
        .bool_ => |arr| {
            std.debug.print("{}", .{arr[row]});
        },
        .string => |arr| {
            std.debug.print("\"{s}\"", .{arr[row]});
        },
    }
}
