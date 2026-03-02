//! SIMD Vector Operations for Fused Code Generation
//!
//! Generates optimized SIMD code for vector operations like cosine distance,
//! L2 distance, and dot product. These are used in the fused query codegen
//! for vectorized similarity search.

const std = @import("std");

/// SIMD operation type
pub const SimdOp = enum {
    cosine_distance,
    l2_distance,
    dot_product,
};

/// Generate SIMD helper function declaration
/// Returns the function name that was generated
pub fn genSimdFunction(
    code: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    op: SimdOp,
    name: []const u8,
) !void {
    switch (op) {
        .cosine_distance => try genCosineDistance(code, allocator, name),
        .l2_distance => try genL2Distance(code, allocator, name),
        .dot_product => try genDotProduct(code, allocator, name),
    }
}

/// Generate cosine distance SIMD function
fn genCosineDistance(code: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8) !void {
    const template =
        \\fn simd_{s}(a: [*]const f32, b: [*]const f32, len: usize) f32 {{
        \\    const Vector = @Vector(8, f32);
        \\    var dot_sum: Vector = @splat(0);
        \\    var norm_a: Vector = @splat(0);
        \\    var norm_b: Vector = @splat(0);
        \\
        \\    var i: usize = 0;
        \\    while (i + 8 <= len) : (i += 8) {{
        \\        const va: Vector = a[i..][0..8].*;
        \\        const vb: Vector = b[i..][0..8].*;
        \\        dot_sum += va * vb;
        \\        norm_a += va * va;
        \\        norm_b += vb * vb;
        \\    }}
        \\
        \\    // Handle remaining elements
        \\    var dot_tail: f32 = 0;
        \\    var na_tail: f32 = 0;
        \\    var nb_tail: f32 = 0;
        \\    while (i < len) : (i += 1) {{
        \\        dot_tail += a[i] * b[i];
        \\        na_tail += a[i] * a[i];
        \\        nb_tail += b[i] * b[i];
        \\    }}
        \\
        \\    // Horizontal sum
        \\    const dot = @reduce(.Add, dot_sum) + dot_tail;
        \\    const na = @sqrt(@reduce(.Add, norm_a) + na_tail);
        \\    const nb = @sqrt(@reduce(.Add, norm_b) + nb_tail);
        \\
        \\    return 1.0 - (dot / (na * nb + 1e-10));
        \\}}
        \\
        \\
    ;

    const writer = code.writer(allocator);
    try writer.print(template, .{name});
}

/// Generate L2 (Euclidean) distance SIMD function
fn genL2Distance(code: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8) !void {
    const template =
        \\fn simd_{s}(a: [*]const f32, b: [*]const f32, len: usize) f32 {{
        \\    const Vector = @Vector(8, f32);
        \\    var sum_sq: Vector = @splat(0);
        \\
        \\    var i: usize = 0;
        \\    while (i + 8 <= len) : (i += 8) {{
        \\        const va: Vector = a[i..][0..8].*;
        \\        const vb: Vector = b[i..][0..8].*;
        \\        const diff = va - vb;
        \\        sum_sq += diff * diff;
        \\    }}
        \\
        \\    // Handle remaining elements
        \\    var tail: f32 = 0;
        \\    while (i < len) : (i += 1) {{
        \\        const diff = a[i] - b[i];
        \\        tail += diff * diff;
        \\    }}
        \\
        \\    return @sqrt(@reduce(.Add, sum_sq) + tail);
        \\}}
        \\
        \\
    ;

    const writer = code.writer(allocator);
    try writer.print(template, .{name});
}

/// Generate dot product SIMD function
fn genDotProduct(code: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8) !void {
    const template =
        \\fn simd_{s}(a: [*]const f32, b: [*]const f32, len: usize) f32 {{
        \\    const Vector = @Vector(8, f32);
        \\    var sum: Vector = @splat(0);
        \\
        \\    var i: usize = 0;
        \\    while (i + 8 <= len) : (i += 8) {{
        \\        const va: Vector = a[i..][0..8].*;
        \\        const vb: Vector = b[i..][0..8].*;
        \\        sum += va * vb;
        \\    }}
        \\
        \\    // Handle remaining elements
        \\    var tail: f32 = 0;
        \\    while (i < len) : (i += 1) {{
        \\        tail += a[i] * b[i];
        \\    }}
        \\
        \\    return @reduce(.Add, sum) + tail;
        \\}}
        \\
        \\
    ;

    const writer = code.writer(allocator);
    try writer.print(template, .{name});
}

/// Detect if a function name is a SIMD vector operation
pub fn detectSimdOp(name: []const u8) ?SimdOp {
    const ops = std.StaticStringMap(SimdOp).initComptime(.{
        .{ "cosine_distance", .cosine_distance },
        .{ "cosine_sim", .cosine_distance },
        .{ "cosine_similarity", .cosine_distance },
        .{ "l2_distance", .l2_distance },
        .{ "euclidean_distance", .l2_distance },
        .{ "dot", .dot_product },
        .{ "dot_product", .dot_product },
    });
    return ops.get(name);
}

// ============================================================================
// Tests
// ============================================================================

test "detect SIMD operations" {
    try std.testing.expect(detectSimdOp("cosine_distance") == .cosine_distance);
    try std.testing.expect(detectSimdOp("l2_distance") == .l2_distance);
    try std.testing.expect(detectSimdOp("dot") == .dot_product);
    try std.testing.expect(detectSimdOp("unknown") == null);
}

test "generate cosine distance function" {
    const allocator = std.testing.allocator;
    var code: std.ArrayList(u8) = .{};
    defer code.deinit(allocator);

    try genSimdFunction(&code, allocator, .cosine_distance, "vec_dist");
    try std.testing.expect(std.mem.indexOf(u8, code.items, "simd_vec_dist") != null);
    try std.testing.expect(std.mem.indexOf(u8, code.items, "@Vector(8, f32)") != null);
}

test "generate l2 distance function" {
    const allocator = std.testing.allocator;
    var code: std.ArrayList(u8) = .{};
    defer code.deinit(allocator);

    try genSimdFunction(&code, allocator, .l2_distance, "l2");
    try std.testing.expect(std.mem.indexOf(u8, code.items, "simd_l2") != null);
    try std.testing.expect(std.mem.indexOf(u8, code.items, "@sqrt") != null);
}

test "generate dot product function" {
    const allocator = std.testing.allocator;
    var code: std.ArrayList(u8) = .{};
    defer code.deinit(allocator);

    try genSimdFunction(&code, allocator, .dot_product, "dot");
    try std.testing.expect(std.mem.indexOf(u8, code.items, "simd_dot") != null);
    try std.testing.expect(std.mem.indexOf(u8, code.items, "@reduce(.Add, sum)") != null);
}
