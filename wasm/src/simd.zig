//! SIMD and Parallel Compute Primitives
//!
//! Provides optimized vector operations with automatic threshold-based dispatch:
//! - Small data: scalar operations
//! - Medium data: SIMD vectorized operations
//! - Large data: parallel + SIMD operations
//!
//! Usage:
//!   const simd = @import("simd.zig");
//!   const norm = simd.l2Norm(data);  // Auto-dispatches based on size

const std = @import("std");

// =============================================================================
// Configuration Constants
// =============================================================================

/// SIMD vector width for f64 (4 x f64 = 256 bits for AVX)
pub const SIMD_WIDTH_F64 = 4;
/// SIMD vector width for f32 (8 x f32 = 256 bits for AVX)
pub const SIMD_WIDTH_F32 = 8;

/// Batch size for cache-friendly processing
pub const VECTOR_SIZE = 1024;

/// Threshold for using SIMD (below this, scalar is fine)
pub const SIMD_THRESHOLD = 64;

/// Threshold for using parallel execution
pub const PARALLEL_THRESHOLD = 20000;

/// Minimum rows per thread to justify parallelization overhead
pub const MIN_ROWS_PER_THREAD = 10000;

/// Maximum threads to use
pub const MAX_THREADS = 16;

// =============================================================================
// SIMD Vector Types
// =============================================================================

pub const Vec4F64 = @Vector(SIMD_WIDTH_F64, f64);
pub const Vec8F32 = @Vector(SIMD_WIDTH_F32, f32);

// =============================================================================
// L2 Norm (Euclidean Length)
// =============================================================================

/// Compute L2 norm with automatic threshold-based dispatch
/// Returns: sqrt(sum(x^2))
pub fn l2Norm(data: []const f64) f64 {
    if (data.len < SIMD_THRESHOLD) {
        return scalarL2Norm(data);
    } else if (data.len < PARALLEL_THRESHOLD) {
        return simdL2Norm(data);
    } else {
        const num_threads = std.Thread.getCpuCount() catch 4;
        return parallelL2Norm(data, num_threads);
    }
}

/// Scalar L2 norm - for small data
pub fn scalarL2Norm(data: []const f64) f64 {
    var acc: f64 = 0;
    for (data) |x| {
        acc += x * x;
    }
    return @sqrt(acc);
}

/// SIMD L2 norm - for medium data
pub fn simdL2Norm(data: []const f64) f64 {
    return @sqrt(simdSumSquares(data));
}

/// SIMD sum of squares (used by L2 norm and other operations)
pub fn simdSumSquares(data: []const f64) f64 {
    const len = data.len;
    var sum_vec: Vec4F64 = @splat(0.0);

    // Process in VECTOR_SIZE batches for cache locality
    var batch_start: usize = 0;
    while (batch_start < len) : (batch_start += VECTOR_SIZE) {
        const batch_end = @min(batch_start + VECTOR_SIZE, len);
        const batch = data[batch_start..batch_end];

        // SIMD loop - process 4 elements at a time
        var i: usize = 0;
        while (i + SIMD_WIDTH_F64 <= batch.len) : (i += SIMD_WIDTH_F64) {
            const vec: Vec4F64 = batch[i..][0..SIMD_WIDTH_F64].*;
            sum_vec += vec * vec;
        }

        // Handle remaining elements
        while (i < batch.len) : (i += 1) {
            sum_vec[0] += batch[i] * batch[i];
        }
    }

    return @reduce(.Add, sum_vec);
}

/// Thread context for parallel L2 norm
const L2ThreadContext = struct {
    data: []const f64,
    result: f64 = 0,
};

fn parallelL2Worker(ctx: *L2ThreadContext) void {
    ctx.result = simdSumSquares(ctx.data);
}

/// Parallel SIMD L2 norm - for large data
pub fn parallelL2Norm(data: []const f64, num_threads: usize) f64 {
    // Validate inputs
    if (data.len < MIN_ROWS_PER_THREAD * 2 or num_threads <= 1) {
        return simdL2Norm(data);
    }

    const actual_threads = @min(@min(num_threads, MAX_THREADS), data.len / MIN_ROWS_PER_THREAD);
    if (actual_threads <= 1) {
        return simdL2Norm(data);
    }

    const chunk_size = data.len / actual_threads;

    var contexts: [MAX_THREADS]L2ThreadContext = undefined;
    var threads: [MAX_THREADS]std.Thread = undefined;
    var spawned: usize = 0;

    // Spawn worker threads
    for (0..actual_threads - 1) |t| {
        const start = t * chunk_size;
        const end = start + chunk_size;
        contexts[t] = .{ .data = data[start..end] };
        threads[t] = std.Thread.spawn(.{}, parallelL2Worker, .{&contexts[t]}) catch {
            contexts[t].result = simdSumSquares(contexts[t].data);
            continue;
        };
        spawned += 1;
    }

    // Process last chunk in main thread
    const last_start = (actual_threads - 1) * chunk_size;
    contexts[actual_threads - 1] = .{ .data = data[last_start..] };
    contexts[actual_threads - 1].result = simdSumSquares(contexts[actual_threads - 1].data);

    // Wait for all threads and sum results
    var total_sum: f64 = contexts[actual_threads - 1].result;
    for (0..spawned) |t| {
        threads[t].join();
        total_sum += contexts[t].result;
    }

    return @sqrt(total_sum);
}

// =============================================================================
// Sum
// =============================================================================

/// Compute sum with automatic threshold-based dispatch
pub fn sum(data: []const f64) f64 {
    if (data.len < SIMD_THRESHOLD) {
        return scalarSum(data);
    } else if (data.len < PARALLEL_THRESHOLD) {
        return simdSum(data);
    } else {
        const num_threads = std.Thread.getCpuCount() catch 4;
        return parallelSum(data, num_threads);
    }
}

pub fn scalarSum(data: []const f64) f64 {
    var total: f64 = 0;
    for (data) |x| {
        total += x;
    }
    return total;
}

pub fn simdSum(data: []const f64) f64 {
    const len = data.len;
    var sum_vec: Vec4F64 = @splat(0.0);

    var batch_start: usize = 0;
    while (batch_start < len) : (batch_start += VECTOR_SIZE) {
        const batch_end = @min(batch_start + VECTOR_SIZE, len);
        const batch = data[batch_start..batch_end];

        var i: usize = 0;
        while (i + SIMD_WIDTH_F64 <= batch.len) : (i += SIMD_WIDTH_F64) {
            const vec: Vec4F64 = batch[i..][0..SIMD_WIDTH_F64].*;
            sum_vec += vec;
        }

        while (i < batch.len) : (i += 1) {
            sum_vec[0] += batch[i];
        }
    }

    return @reduce(.Add, sum_vec);
}

const SumThreadContext = struct {
    data: []const f64,
    result: f64 = 0,
};

fn parallelSumWorker(ctx: *SumThreadContext) void {
    ctx.result = simdSum(ctx.data);
}

pub fn parallelSum(data: []const f64, num_threads: usize) f64 {
    if (data.len < MIN_ROWS_PER_THREAD * 2 or num_threads <= 1) {
        return simdSum(data);
    }

    const actual_threads = @min(@min(num_threads, MAX_THREADS), data.len / MIN_ROWS_PER_THREAD);
    if (actual_threads <= 1) {
        return simdSum(data);
    }

    const chunk_size = data.len / actual_threads;

    var contexts: [MAX_THREADS]SumThreadContext = undefined;
    var threads: [MAX_THREADS]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..actual_threads - 1) |t| {
        const start = t * chunk_size;
        const end = start + chunk_size;
        contexts[t] = .{ .data = data[start..end] };
        threads[t] = std.Thread.spawn(.{}, parallelSumWorker, .{&contexts[t]}) catch {
            contexts[t].result = simdSum(contexts[t].data);
            continue;
        };
        spawned += 1;
    }

    const last_start = (actual_threads - 1) * chunk_size;
    contexts[actual_threads - 1] = .{ .data = data[last_start..] };
    contexts[actual_threads - 1].result = simdSum(contexts[actual_threads - 1].data);

    var total: f64 = contexts[actual_threads - 1].result;
    for (0..spawned) |t| {
        threads[t].join();
        total += contexts[t].result;
    }

    return total;
}

// =============================================================================
// Dot Product
// =============================================================================

/// Compute dot product with automatic threshold-based dispatch
pub fn dotProduct(a: []const f64, b: []const f64) f64 {
    const len = @min(a.len, b.len);
    if (len < SIMD_THRESHOLD) {
        return scalarDotProduct(a[0..len], b[0..len]);
    } else {
        return simdDotProduct(a[0..len], b[0..len]);
    }
}

pub fn scalarDotProduct(a: []const f64, b: []const f64) f64 {
    var acc: f64 = 0;
    for (a, b) |x, y| {
        acc += x * y;
    }
    return acc;
}

pub fn simdDotProduct(a: []const f64, b: []const f64) f64 {
    const len = @min(a.len, b.len);
    var sum_vec: Vec4F64 = @splat(0.0);

    var i: usize = 0;
    while (i + SIMD_WIDTH_F64 <= len) : (i += SIMD_WIDTH_F64) {
        const vec_a: Vec4F64 = a[i..][0..SIMD_WIDTH_F64].*;
        const vec_b: Vec4F64 = b[i..][0..SIMD_WIDTH_F64].*;
        sum_vec += vec_a * vec_b;
    }

    var result: f64 = @reduce(.Add, sum_vec);

    // Handle remaining elements
    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

// =============================================================================
// Batch Dot Product (for embeddings)
// =============================================================================

/// Batch dot product for f32 matrix against f64 query vector
/// matrix: [num_rows x dim] flattened, query: [dim], out: [num_rows]
pub fn batchDotProductF32(
    matrix: []const f32,
    query: []const f64,
    dim: usize,
    out: []f64,
) void {
    const num_rows = @min(matrix.len / dim, out.len);

    // Pre-convert query to f32 once
    var query_f32: [1024]f32 = undefined; // Max dim 1024
    const query_dim = @min(dim, 1024);
    for (0..query_dim) |i| {
        query_f32[i] = @floatCast(query[i]);
    }

    // Process each row with SIMD
    for (0..num_rows) |row| {
        const row_start = row * dim;
        var sum_vec: Vec8F32 = @splat(0.0);

        var i: usize = 0;
        while (i + SIMD_WIDTH_F32 <= dim) : (i += SIMD_WIDTH_F32) {
            const mat_vec: Vec8F32 = matrix[row_start + i ..][0..SIMD_WIDTH_F32].*;
            const q_vec: Vec8F32 = query_f32[i..][0..SIMD_WIDTH_F32].*;
            sum_vec += mat_vec * q_vec;
        }

        var score: f32 = @reduce(.Add, sum_vec);

        while (i < dim) : (i += 1) {
            score += matrix[row_start + i] * query_f32[i];
        }

        out[row] = @floatCast(score);
    }
}

// =============================================================================
// Filter Count
// =============================================================================

/// Count elements greater than threshold with auto dispatch
pub fn countGreaterThan(data: []const f64, threshold: f64) u64 {
    if (data.len < SIMD_THRESHOLD) {
        return scalarCountGreaterThan(data, threshold);
    } else if (data.len < PARALLEL_THRESHOLD) {
        return simdCountGreaterThan(data, threshold);
    } else {
        const num_threads = std.Thread.getCpuCount() catch 4;
        return parallelCountGreaterThan(data, threshold, num_threads);
    }
}

pub fn scalarCountGreaterThan(data: []const f64, threshold: f64) u64 {
    var count: u64 = 0;
    for (data) |x| {
        if (x > threshold) count += 1;
    }
    return count;
}

pub fn simdCountGreaterThan(data: []const f64, threshold: f64) u64 {
    const len = data.len;
    const thresh_vec: Vec4F64 = @splat(threshold);
    var count: u64 = 0;

    var batch_start: usize = 0;
    while (batch_start < len) : (batch_start += VECTOR_SIZE) {
        const batch_end = @min(batch_start + VECTOR_SIZE, len);
        const batch = data[batch_start..batch_end];

        var i: usize = 0;
        while (i + SIMD_WIDTH_F64 <= batch.len) : (i += SIMD_WIDTH_F64) {
            const vec: Vec4F64 = batch[i..][0..SIMD_WIDTH_F64].*;
            const mask = vec > thresh_vec;
            count += @popCount(@as(u4, @bitCast(mask)));
        }

        while (i < batch.len) : (i += 1) {
            if (batch[i] > threshold) count += 1;
        }
    }

    return count;
}

const FilterThreadContext = struct {
    data: []const f64,
    threshold: f64,
    result: u64 = 0,
};

fn parallelFilterWorker(ctx: *FilterThreadContext) void {
    ctx.result = simdCountGreaterThan(ctx.data, ctx.threshold);
}

pub fn parallelCountGreaterThan(data: []const f64, threshold: f64, num_threads: usize) u64 {
    if (data.len < MIN_ROWS_PER_THREAD * 2 or num_threads <= 1) {
        return simdCountGreaterThan(data, threshold);
    }

    const actual_threads = @min(@min(num_threads, MAX_THREADS), data.len / MIN_ROWS_PER_THREAD);
    if (actual_threads <= 1) {
        return simdCountGreaterThan(data, threshold);
    }

    const chunk_size = data.len / actual_threads;

    var contexts: [MAX_THREADS]FilterThreadContext = undefined;
    var threads: [MAX_THREADS]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..actual_threads - 1) |t| {
        const start = t * chunk_size;
        const end = start + chunk_size;
        contexts[t] = .{ .data = data[start..end], .threshold = threshold };
        threads[t] = std.Thread.spawn(.{}, parallelFilterWorker, .{&contexts[t]}) catch {
            contexts[t].result = simdCountGreaterThan(contexts[t].data, threshold);
            continue;
        };
        spawned += 1;
    }

    const last_start = (actual_threads - 1) * chunk_size;
    contexts[actual_threads - 1] = .{ .data = data[last_start..], .threshold = threshold };
    contexts[actual_threads - 1].result = simdCountGreaterThan(contexts[actual_threads - 1].data, threshold);

    var total: u64 = contexts[actual_threads - 1].result;
    for (0..spawned) |t| {
        threads[t].join();
        total += contexts[t].result;
    }

    return total;
}

// =============================================================================
// Tests
// =============================================================================

test "l2Norm" {
    const data = [_]f64{ 3.0, 4.0 };
    const result = l2Norm(&data);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result, 0.0001);
}

test "sum" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const result = sum(&data);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), result, 0.0001);
}

test "dotProduct" {
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    const b = [_]f64{ 4.0, 5.0, 6.0 };
    const result = dotProduct(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f64, 32.0), result, 0.0001);
}

test "countGreaterThan" {
    const data = [_]f64{ 1.0, 5.0, 3.0, 7.0, 2.0 };
    const result = countGreaterThan(&data, 3.0);
    try std.testing.expectEqual(@as(u64, 2), result);
}
