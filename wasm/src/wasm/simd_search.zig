//! SIMD-Optimized Vector Search for WASM
//!
//! Provides SIMD-accelerated similarity search operations using
//! WASM SIMD 128-bit vectors (4 x f32).

const std = @import("std");

// Import from parent module - need ColumnBuffer and getColumnBuffer
const wasm = @import("../wasm.zig");
pub const ColumnBuffer = wasm.ColumnBuffer;
pub const getColumnBuffer = wasm.getColumnBuffer;

// ============================================================================
// SIMD Types
// ============================================================================

/// WASM SIMD 128-bit vector (4 x f32)
const Vec4 = @Vector(4, f32);

// ============================================================================
// SIMD Helper Functions
// ============================================================================

/// SIMD dot product for f32 vectors (4-wide)
pub fn simdDotProduct(a_ptr: [*]const f32, b_ptr: [*]const f32, dim: usize) f32 {
    var sum: Vec4 = @splat(0);
    var i: usize = 0;

    // Process 4 elements at a time
    while (i + 4 <= dim) : (i += 4) {
        const va: Vec4 = .{ a_ptr[i], a_ptr[i + 1], a_ptr[i + 2], a_ptr[i + 3] };
        const vb: Vec4 = .{ b_ptr[i], b_ptr[i + 1], b_ptr[i + 2], b_ptr[i + 3] };
        sum += va * vb;
    }

    // Horizontal sum
    var result = @reduce(.Add, sum);

    // Handle remainder
    while (i < dim) : (i += 1) {
        result += a_ptr[i] * b_ptr[i];
    }

    return result;
}

/// SIMD L2 norm squared
pub fn simdNormSquared(ptr: [*]const f32, dim: usize) f32 {
    var sum: Vec4 = @splat(0);
    var i: usize = 0;

    while (i + 4 <= dim) : (i += 4) {
        const v: Vec4 = .{ ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3] };
        sum += v * v;
    }

    var result = @reduce(.Add, sum);

    while (i < dim) : (i += 1) {
        result += ptr[i] * ptr[i];
    }

    return result;
}

/// Find top-k from pre-computed scores
/// Uses partial selection for better performance than full sort
fn findTopK(
    scores: [*]const f32,
    num_scores: usize,
    top_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
) usize {
    const actual_k = @min(top_k, num_scores);

    // Initialize with worst scores
    for (0..actual_k) |i| {
        out_indices[i] = 0;
        out_scores[i] = -2.0;
    }

    // Simple insertion sort into top-k (good for small k)
    for (0..num_scores) |i| {
        const score = scores[i];

        if (score > out_scores[actual_k - 1]) {
            // Find insertion point
            var insert_pos: usize = actual_k - 1;
            while (insert_pos > 0 and score > out_scores[insert_pos - 1]) {
                insert_pos -= 1;
            }

            // Shift elements down
            var j: usize = actual_k - 1;
            while (j > insert_pos) {
                out_indices[j] = out_indices[j - 1];
                out_scores[j] = out_scores[j - 1];
                j -= 1;
            }

            // Insert
            out_indices[insert_pos] = @intCast(i);
            out_scores[insert_pos] = score;
        }
    }

    return actual_k;
}

// ============================================================================
// WASM Exports
// ============================================================================

/// SIMD cosine similarity for pre-normalized vectors (just dot product)
pub export fn simdCosineSimilarityNormalized(
    vec_a: [*]const f32,
    vec_b: [*]const f32,
    dim: usize,
) f32 {
    return simdDotProduct(vec_a, vec_b, dim);
}

/// SIMD cosine similarity for un-normalized vectors
pub export fn simdCosineSimilarity(
    vec_a: [*]const f32,
    vec_b: [*]const f32,
    dim: usize,
) f32 {
    const dot = simdDotProduct(vec_a, vec_b, dim);
    const norm_a = @sqrt(simdNormSquared(vec_a, dim));
    const norm_b = @sqrt(simdNormSquared(vec_b, dim));
    const denom = norm_a * norm_b;
    if (denom == 0) return 0;
    return dot / denom;
}

/// Batch compute similarities between query and multiple vectors
/// Much faster than calling simdCosineSimilarity in a loop
/// vectors_ptr: flattened array of [num_vectors * dim] f32
/// out_scores: array of [num_vectors] f32
pub export fn batchCosineSimilarity(
    query_ptr: [*]const f32,
    vectors_ptr: [*]const f32,
    dim: usize,
    num_vectors: usize,
    out_scores: [*]f32,
    normalized: u32,
) void {
    // Pre-compute query norm if not normalized
    const query_norm = if (normalized == 0) @sqrt(simdNormSquared(query_ptr, dim)) else 1.0;

    for (0..num_vectors) |i| {
        const vec_ptr = vectors_ptr + i * dim;
        const dot = simdDotProduct(query_ptr, vec_ptr, dim);

        if (normalized != 0) {
            // Vectors are L2-normalized, dot product = cosine similarity
            out_scores[i] = dot;
        } else {
            const vec_norm = @sqrt(simdNormSquared(vec_ptr, dim));
            const denom = query_norm * vec_norm;
            out_scores[i] = if (denom == 0) 0 else dot / denom;
        }
    }
}

/// Find top-k most similar vectors to query (SIMD optimized)
/// Returns number of results written
/// out_indices: row indices of top matches
/// out_scores: similarity scores
pub export fn vectorSearchTopK(
    col_idx: u32,
    query_ptr: [*]const f32,
    query_dim: usize,
    top_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    if (buf.rows == 0) return 0;

    const dim = buf.size / (buf.rows * 4);
    if (dim != query_dim) return 0;

    const actual_k = @min(top_k, buf.rows);

    // Initialize with worst scores
    for (0..actual_k) |i| {
        out_indices[i] = 0;
        out_scores[i] = -2.0;
    }

    // Pre-compute query norm (assume vectors may not be normalized)
    const query_norm = @sqrt(simdNormSquared(query_ptr, query_dim));

    // Scan all vectors using SIMD
    for (0..buf.rows) |row| {
        const vec_start = buf.start + row * dim * 4;
        const vec_ptr: [*]const f32 = @ptrCast(@alignCast(buf.data.ptr + vec_start));

        const dot = simdDotProduct(query_ptr, vec_ptr, dim);
        const vec_norm = @sqrt(simdNormSquared(vec_ptr, dim));
        const denom = query_norm * vec_norm;
        const score: f32 = if (denom == 0) 0 else dot / denom;

        // Insert into top-k if better than worst
        if (score > out_scores[actual_k - 1]) {
            var insert_pos: usize = actual_k - 1;
            while (insert_pos > 0 and score > out_scores[insert_pos - 1]) insert_pos -= 1;

            var j: usize = actual_k - 1;
            while (j > insert_pos) {
                out_indices[j] = out_indices[j - 1];
                out_scores[j] = out_scores[j - 1];
                j -= 1;
            }
            out_indices[insert_pos] = @intCast(row);
            out_scores[insert_pos] = score;
        }
    }
    return actual_k;
}

/// Vector search on raw buffer (for worker-based processing)
/// Searches vectors directly from a provided buffer, not from file_data
/// normalized: 1 if vectors are L2-normalized (skip norm computation)
pub export fn vectorSearchBuffer(
    vectors_ptr: [*]const f32,
    num_vectors: usize,
    dim: usize,
    query_ptr: [*]const f32,
    top_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
    normalized: u32,
    start_index: u32,
) usize {
    if (num_vectors == 0 or top_k == 0) return 0;
    const actual_k = @min(top_k, num_vectors);

    // Initialize with worst scores ONLY if this is the start of a search (start_index == 0)
    if (start_index == 0) {
        for (0..top_k) |i| {
            out_indices[i] = 0;
            out_scores[i] = -2.0;
        }
    }

    // Pre-compute query norm if not normalized
    const query_norm = if (normalized == 0) @sqrt(simdNormSquared(query_ptr, dim)) else 1.0;

    // Scan all vectors using SIMD
    for (0..num_vectors) |row| {
        const vec_ptr = vectors_ptr + row * dim;

        const dot = simdDotProduct(query_ptr, vec_ptr, dim);

        var score: f32 = undefined;
        if (normalized != 0) {
            // For L2-normalized vectors, dot product = cosine similarity
            score = dot;
        } else {
            const vec_norm = @sqrt(simdNormSquared(vec_ptr, dim));
            const denom = query_norm * vec_norm;
            score = if (denom == 0) 0 else dot / denom;
        }

        // Insert into top-k if better than worst (and we have a valid k)
        if (actual_k > 0 and score > out_scores[top_k - 1]) {
            var insert_pos: usize = top_k - 1;
            while (insert_pos > 0 and score > out_scores[insert_pos - 1]) {
                insert_pos -= 1;
            }

            var j: usize = top_k - 1;
            while (j > insert_pos) {
                out_indices[j] = out_indices[j - 1];
                out_scores[j] = out_scores[j - 1];
                j -= 1;
            }

            // Store global index (start_index + local row)
            out_indices[insert_pos] = start_index + @as(u32, @intCast(row));
            out_scores[insert_pos] = score;
        }
    }

    return actual_k;
}

/// Merge multiple top-k results into final top-k
/// Used by main thread to combine results from workers
pub export fn mergeTopK(
    indices_arrays: [*]const [*]const u32,
    scores_arrays: [*]const [*]const f32,
    num_arrays: usize,
    k_per_array: usize,
    final_k: usize,
    out_indices: [*]u32,
    out_scores: [*]f32,
) usize {
    const actual_k = @min(final_k, num_arrays * k_per_array);

    // Initialize
    for (0..actual_k) |i| {
        out_indices[i] = 0;
        out_scores[i] = -2.0;
    }

    // Merge all results
    for (0..num_arrays) |arr_idx| {
        const indices = indices_arrays[arr_idx];
        const scores = scores_arrays[arr_idx];

        for (0..k_per_array) |i| {
            const score = scores[i];
            if (score <= -2.0) continue; // Skip invalid

            if (score > out_scores[actual_k - 1]) {
                var insert_pos: usize = actual_k - 1;
                while (insert_pos > 0 and score > out_scores[insert_pos - 1]) {
                    insert_pos -= 1;
                }

                var j: usize = actual_k - 1;
                while (j > insert_pos) {
                    out_indices[j] = out_indices[j - 1];
                    out_scores[j] = out_scores[j - 1];
                    j -= 1;
                }

                out_indices[insert_pos] = indices[i];
                out_scores[insert_pos] = score;
            }
        }
    }

    return actual_k;
}

// ============================================================================
// Tests
// ============================================================================

test "simd_search: simdDotProduct" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const result = simdDotProduct(&a, &b, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result, 0.0001);
}

test "simd_search: simdNormSquared" {
    const v = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
    const result = simdNormSquared(&v, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), result, 0.0001);
}

test "simd_search: simdCosineSimilarity identical" {
    const v = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const result = simdCosineSimilarity(&v, &v, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.0001);
}

test "simd_search: simdCosineSimilarity orthogonal" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const result = simdCosineSimilarity(&a, &b, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.0001);
}

test "simd_search: simdCosineSimilarityNormalized" {
    // Pre-normalized unit vectors
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.707107, 0.707107, 0.0, 0.0 }; // 45 degrees
    const result = simdCosineSimilarityNormalized(&a, &b, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.707107), result, 0.0001);
}

test "simd_search: findTopK" {
    const scores = [_]f32{ 0.1, 0.9, 0.5, 0.3, 0.7, 0.2 };
    var out_indices: [3]u32 = undefined;
    var out_scores: [3]f32 = undefined;

    const k = findTopK(&scores, 6, 3, &out_indices, &out_scores);
    try std.testing.expectEqual(@as(usize, 3), k);
    try std.testing.expectEqual(@as(u32, 1), out_indices[0]); // 0.9
    try std.testing.expectEqual(@as(u32, 4), out_indices[1]); // 0.7
    try std.testing.expectEqual(@as(u32, 2), out_indices[2]); // 0.5
}
