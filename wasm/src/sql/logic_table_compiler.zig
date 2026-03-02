//! Logic Table Method Extractor - Extracts method bodies from generated Zig
//!
//! This module extracts method bodies from metal0's generated Zig code
//! and transforms them for inlining into fused query functions.
//!
//! The actual Python→Zig compilation happens in metal0_jit.zig.
//! This module focuses on:
//! 1. Parsing the generated Zig source
//! 2. Extracting specific method bodies
//! 3. Transforming param references to column array accesses
//!
//! Usage:
//!   var extractor = MethodExtractor.init(allocator);
//!   defer extractor.deinit();
//!
//!   const method = try extractor.extractMethod(
//!       generated_zig_source,
//!       "FraudDetector",
//!       "risk_score",
//!   );
//!   // method.inlined_body contains Zig code ready for fused loop

const std = @import("std");

// ============================================================================
// Types
// ============================================================================

/// Column types for method return types
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
    vec_f32,
    vec_f64,
    timestamp,
    date,
    decimal,
    unknown,
};

/// Extracted method ready for inlining
pub const ExtractedMethod = struct {
    /// Method name
    name: []const u8,
    /// Class name
    class_name: []const u8,
    /// Method parameters
    params: []const ParamInfo,
    /// Return type
    return_type: ColumnType,
    /// Raw method body (between { and })
    raw_body: []const u8,
    /// Body adapted for fused loop (columns.X[i] access)
    inlined_body: []const u8,
    /// Column dependencies (params that need to be loaded)
    column_deps: []const []const u8,
};

/// Method parameter info
pub const ParamInfo = struct {
    name: []const u8,
    col_type: ColumnType,
};

/// Extractor errors
pub const ExtractorError = error{
    OutOfMemory,
    ClassNotFound,
    MethodNotFound,
    MalformedSignature,
    MalformedMethod,
};

// ============================================================================
// MethodExtractor
// ============================================================================

/// Extracts method bodies from metal0-generated Zig code
pub const MethodExtractor = struct {
    allocator: std.mem.Allocator,
    /// Statistics
    stats: ExtractorStats,

    const Self = @This();

    pub const ExtractorStats = struct {
        methods_extracted: u64 = 0,
        extraction_time_ns: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .stats = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
        // No cleanup needed - caller owns the memory
    }

    /// Extract a specific method from generated Zig source
    pub fn extractMethod(
        self: *Self,
        zig_source: []const u8,
        class_name: []const u8,
        method_name: []const u8,
    ) ExtractorError!ExtractedMethod {
        const start_time = std.time.nanoTimestamp();

        // Find the struct
        const struct_marker = std.fmt.allocPrint(
            self.allocator,
            "pub const {s} = struct {{",
            .{class_name},
        ) catch return ExtractorError.OutOfMemory;
        defer self.allocator.free(struct_marker);

        const struct_start = std.mem.indexOf(u8, zig_source, struct_marker) orelse
            return ExtractorError.ClassNotFound;

        // Find the method within the struct
        const fn_marker = std.fmt.allocPrint(
            self.allocator,
            "pub fn {s}(",
            .{method_name},
        ) catch return ExtractorError.OutOfMemory;
        defer self.allocator.free(fn_marker);

        const method_start = std.mem.indexOfPos(u8, zig_source, struct_start, fn_marker) orelse
            return ExtractorError.MethodNotFound;

        // Extract the method body
        const result = try self.extractMethodBody(zig_source, class_name, method_name, method_start);

        const end_time = std.time.nanoTimestamp();
        self.stats.extraction_time_ns += @intCast(@as(i128, end_time - start_time));
        self.stats.methods_extracted += 1;

        return result;
    }

    /// Auto-detect class name from generated Zig
    pub fn extractClassName(self: *Self, zig_source: []const u8) ExtractorError![]const u8 {
        // Look for: pub const __logic_table__ = true;
        const marker = "pub const __logic_table__ = true;";
        const marker_pos = std.mem.indexOf(u8, zig_source, marker) orelse
            return ExtractorError.ClassNotFound;

        // Search backward for "pub const X = struct {"
        const before = zig_source[0..marker_pos];
        const struct_marker = "= struct {";
        const struct_pos = std.mem.lastIndexOf(u8, before, struct_marker) orelse
            return ExtractorError.ClassNotFound;

        // Find "pub const " before the struct
        const const_marker = "pub const ";
        var search_start: usize = 0;
        var last_const: ?usize = null;

        while (std.mem.indexOfPos(u8, before, search_start, const_marker)) |pos| {
            if (pos < struct_pos) {
                last_const = pos;
                search_start = pos + const_marker.len;
            } else break;
        }

        const const_pos = last_const orelse return ExtractorError.ClassNotFound;
        const name_start = const_pos + const_marker.len;

        // Find end of class name (space before "=")
        const name_end = std.mem.indexOfPos(u8, zig_source, name_start, " ") orelse
            return ExtractorError.MalformedSignature;

        return self.allocator.dupe(u8, zig_source[name_start..name_end]) catch
            return ExtractorError.OutOfMemory;
    }

    /// Extract method body from generated Zig source
    fn extractMethodBody(
        self: *Self,
        zig_source: []const u8,
        class_name: []const u8,
        method_name: []const u8,
        method_start: usize,
    ) ExtractorError!ExtractedMethod {
        // Find opening brace after return type
        const sig_end = std.mem.indexOfPos(u8, zig_source, method_start, ") ") orelse
            std.mem.indexOfPos(u8, zig_source, method_start, ")!") orelse
            return ExtractorError.MalformedSignature;

        const brace_start = std.mem.indexOfPos(u8, zig_source, sig_end, "{") orelse
            return ExtractorError.MalformedMethod;

        // Find matching closing brace
        const brace_end = findMatchingBrace(zig_source, brace_start) orelse
            return ExtractorError.MalformedMethod;

        // Extract body (excluding braces)
        const raw_body = self.allocator.dupe(u8, zig_source[brace_start + 1 .. brace_end]) catch
            return ExtractorError.OutOfMemory;

        // Parse parameters
        const params = try self.parseParams(zig_source, method_start);

        // Parse return type
        const return_type = self.parseReturnType(zig_source, method_start, sig_end);

        // Transform for inlining
        const inlined_body = try self.adaptForInlining(raw_body, params);

        // Extract column dependencies from params
        var column_deps = std.ArrayList([]const u8).init(self.allocator);
        defer column_deps.deinit();
        for (params) |p| {
            if (!std.mem.eql(u8, p.name, "self")) {
                column_deps.append(p.name) catch return ExtractorError.OutOfMemory;
            }
        }

        return ExtractedMethod{
            .name = method_name,
            .class_name = class_name,
            .params = params,
            .return_type = return_type,
            .raw_body = raw_body,
            .inlined_body = inlined_body,
            .column_deps = column_deps.toOwnedSlice() catch return ExtractorError.OutOfMemory,
        };
    }

    /// Parse method parameters
    fn parseParams(self: *Self, zig_source: []const u8, method_start: usize) ExtractorError![]const ParamInfo {
        // Find parameter list between ( and )
        const paren_start = std.mem.indexOfPos(u8, zig_source, method_start, "(") orelse
            return ExtractorError.MalformedSignature;
        const paren_end = std.mem.indexOfPos(u8, zig_source, paren_start, ")") orelse
            return ExtractorError.MalformedSignature;

        const params_str = zig_source[paren_start + 1 .. paren_end];

        var params = std.ArrayList(ParamInfo).init(self.allocator);
        errdefer params.deinit();

        // Split by comma and parse each param
        var param_iter = std.mem.splitScalar(u8, params_str, ',');
        while (param_iter.next()) |param| {
            const trimmed = std.mem.trim(u8, param, " \t\n");
            if (trimmed.len == 0) continue;

            // Skip self parameter
            if (std.mem.startsWith(u8, trimmed, "self")) continue;
            if (std.mem.startsWith(u8, trimmed, "_")) continue;

            // Parse "name: type" or "name: anytype"
            const colon_pos = std.mem.indexOf(u8, trimmed, ":") orelse continue;
            const name = std.mem.trim(u8, trimmed[0..colon_pos], " ");
            const type_str = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " ");

            const col_type = zigTypeToColumnType(type_str);

            params.append(.{
                .name = name,
                .col_type = col_type,
            }) catch return ExtractorError.OutOfMemory;
        }

        return params.toOwnedSlice() catch return ExtractorError.OutOfMemory;
    }

    /// Parse return type from method signature
    fn parseReturnType(
        self: *Self,
        zig_source: []const u8,
        method_start: usize,
        sig_end: usize,
    ) ColumnType {
        _ = self;
        _ = method_start;

        // Find return type after ) and before {
        const after_paren = zig_source[sig_end..];

        // Look for type after "!" or directly after ")"
        if (std.mem.indexOf(u8, after_paren, "!")) |bang_pos| {
            const type_start = bang_pos + 1;
            const type_end = std.mem.indexOf(u8, after_paren[type_start..], " ") orelse
                std.mem.indexOf(u8, after_paren[type_start..], "{") orelse
                return .f64;

            const type_str = std.mem.trim(u8, after_paren[type_start .. type_start + type_end], " \t\n");
            return zigTypeToColumnType(type_str);
        }

        // No error union, look for direct type
        return .f64; // Default to f64
    }

    /// Transform parameter references to column array accesses
    fn adaptForInlining(
        self: *Self,
        raw_body: []const u8,
        params: []const ParamInfo,
    ) ExtractorError![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < raw_body.len) {
            var found_param = false;

            // Check if current position matches any param name
            for (params) |param| {
                if (std.mem.eql(u8, param.name, "self")) continue;

                if (i + param.name.len <= raw_body.len and
                    std.mem.eql(u8, raw_body[i .. i + param.name.len], param.name))
                {
                    // Check it's not part of a larger identifier
                    const before_ok = i == 0 or !isIdentChar(raw_body[i - 1]);
                    const after_ok = i + param.name.len >= raw_body.len or
                        !isIdentChar(raw_body[i + param.name.len]);

                    if (before_ok and after_ok) {
                        // Replace with column access
                        result.writer().print("columns.{s}[i]", .{param.name}) catch
                            return ExtractorError.OutOfMemory;
                        i += param.name.len;
                        found_param = true;
                        break;
                    }
                }
            }

            if (!found_param) {
                // Clean up runtime.xxx calls
                if (std.mem.startsWith(u8, raw_body[i..], "runtime.addNum(")) {
                    // runtime.addNum(a, b) -> (a + b)
                    i += "runtime.addNum(".len;
                    result.append('(') catch return ExtractorError.OutOfMemory;
                    // Copy until comma
                    while (i < raw_body.len and raw_body[i] != ',') : (i += 1) {
                        result.append(raw_body[i]) catch return ExtractorError.OutOfMemory;
                    }
                    result.appendSlice(" + ") catch return ExtractorError.OutOfMemory;
                    i += 2; // Skip ", "
                    // Copy until closing paren
                    while (i < raw_body.len and raw_body[i] != ')') : (i += 1) {
                        result.append(raw_body[i]) catch return ExtractorError.OutOfMemory;
                    }
                    result.append(')') catch return ExtractorError.OutOfMemory;
                    i += 1; // Skip )
                } else if (std.mem.startsWith(u8, raw_body[i..], "runtime.builtinLen(")) {
                    // runtime.builtinLen(x) -> x.len
                    i += "runtime.builtinLen(".len;
                    const end = std.mem.indexOfPos(u8, raw_body, i, ")") orelse raw_body.len;
                    result.appendSlice(raw_body[i..end]) catch return ExtractorError.OutOfMemory;
                    result.appendSlice(".len") catch return ExtractorError.OutOfMemory;
                    i = end + 1;
                } else {
                    result.append(raw_body[i]) catch return ExtractorError.OutOfMemory;
                    i += 1;
                }
            }
        }

        return result.toOwnedSlice() catch return ExtractorError.OutOfMemory;
    }

    /// Get extractor statistics
    pub fn getStats(self: *const Self) ExtractorStats {
        return self.stats;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Find matching closing brace
fn findMatchingBrace(source: []const u8, open_pos: usize) ?usize {
    var depth: usize = 1;
    var i = open_pos + 1;

    while (i < source.len) : (i += 1) {
        if (source[i] == '{') {
            depth += 1;
        } else if (source[i] == '}') {
            depth -= 1;
            if (depth == 0) return i;
        }
    }

    return null;
}

/// Check if character is part of an identifier
fn isIdentChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or
        (c >= 'A' and c <= 'Z') or
        (c >= '0' and c <= '9') or
        c == '_';
}

/// Convert Zig type string to ColumnType
fn zigTypeToColumnType(type_str: []const u8) ColumnType {
    if (std.mem.eql(u8, type_str, "f64")) return .f64;
    if (std.mem.eql(u8, type_str, "f32")) return .f32;
    if (std.mem.eql(u8, type_str, "i64") or std.mem.eql(u8, type_str, "isize")) return .i64;
    if (std.mem.eql(u8, type_str, "i32")) return .i32;
    if (std.mem.eql(u8, type_str, "i16")) return .i16;
    if (std.mem.eql(u8, type_str, "i8")) return .i8;
    if (std.mem.eql(u8, type_str, "u64") or std.mem.eql(u8, type_str, "usize")) return .u64;
    if (std.mem.eql(u8, type_str, "u32")) return .u32;
    if (std.mem.eql(u8, type_str, "u16")) return .u16;
    if (std.mem.eql(u8, type_str, "u8")) return .u8;
    if (std.mem.eql(u8, type_str, "bool")) return .bool;
    if (std.mem.startsWith(u8, type_str, "[]const u8")) return .string;
    if (std.mem.startsWith(u8, type_str, "[]const f32")) return .vec_f32;
    if (std.mem.startsWith(u8, type_str, "[]const f64")) return .vec_f64;
    if (std.mem.eql(u8, type_str, "anytype")) return .f64; // Default for generic
    return .unknown;
}

// ============================================================================
// Tests
// ============================================================================

test "MethodExtractor init/deinit" {
    const allocator = std.testing.allocator;
    var extractor = MethodExtractor.init(allocator);
    defer extractor.deinit();
}

test "findMatchingBrace" {
    const source = "fn foo() { if (x) { y; } }";
    const open_pos = std.mem.indexOf(u8, source, "{").?;
    const close_pos = findMatchingBrace(source, open_pos).?;
    try std.testing.expectEqual(@as(usize, source.len - 1), close_pos);
}

test "isIdentChar" {
    try std.testing.expect(isIdentChar('a'));
    try std.testing.expect(isIdentChar('Z'));
    try std.testing.expect(isIdentChar('5'));
    try std.testing.expect(isIdentChar('_'));
    try std.testing.expect(!isIdentChar(' '));
    try std.testing.expect(!isIdentChar('.'));
}

test "zigTypeToColumnType" {
    try std.testing.expectEqual(ColumnType.f64, zigTypeToColumnType("f64"));
    try std.testing.expectEqual(ColumnType.i64, zigTypeToColumnType("i64"));
    try std.testing.expectEqual(ColumnType.string, zigTypeToColumnType("[]const u8"));
}

test "extractMethod basic" {
    const allocator = std.testing.allocator;
    var extractor = MethodExtractor.init(allocator);
    defer extractor.deinit();

    const zig_source =
        \\pub const TestClass = struct {
        \\    pub const __logic_table__ = true;
        \\
        \\    pub fn init(_: std.mem.Allocator) !@This() { return .{}; }
        \\
        \\    pub fn score(self: *const @This(), amount: f64) !f64 {
        \\        _ = self;
        \\        return amount * 0.5;
        \\    }
        \\};
    ;

    const method = try extractor.extractMethod(zig_source, "TestClass", "score");
    try std.testing.expectEqualStrings("score", method.name);
    try std.testing.expectEqual(ColumnType.f64, method.return_type);

    // Check that raw_body was extracted
    try std.testing.expect(method.raw_body.len > 0);

    // Check that inlined_body contains column access pattern
    try std.testing.expect(std.mem.indexOf(u8, method.inlined_body, "columns.amount[i]") != null);

    // Cleanup
    allocator.free(method.raw_body);
    allocator.free(method.inlined_body);
    allocator.free(method.params);
    allocator.free(method.column_deps);
}
