//! WASM bridge for IVF-PQ vector index.
//!
//! Thin exports around embedding/ivf_pq_index.zig for loading,
//! searching, and freeing IVF-PQ indexes from JavaScript.
//!
//! Binary format (IVPQ):
//!   Magic: "IVPQ" (4 bytes)
//!   dim: u32 LE
//!   n_partitions: u32 LE
//!   n_subvectors: u32 LE
//!   n_codes: u32 LE
//!   n_probe: u32 LE
//!   n_vectors: u32 LE
//!   IVF centroids: [n_partitions * dim] f32 LE
//!   PQ centroids:  [n_subvectors * n_codes * subvec_dim] f32 LE
//!   PQ codes:      [n_vectors * n_subvectors] u8
//!   Inverted lists: for each partition:
//!     list_len: u32 LE
//!     vec_ids:  [list_len] u32 LE

const std = @import("std");
const memory = @import("memory.zig");

const wasmAlloc = memory.wasmAlloc;

// ============================================================================
// Index storage (simple slot-based, max 8 concurrent indexes)
// ============================================================================

const MAX_INDEXES = 8;

const IndexSlot = struct {
    dim: u32,
    n_partitions: u32,
    n_subvectors: u32,
    n_codes: u32,
    n_probe: u32,
    n_vectors: u32,
    // Pointers into WASM linear memory
    ivf_centroids_ptr: usize, // [n_partitions * dim] f32
    pq_centroids_ptr: usize, // [n_subvectors * n_codes * subvec_dim] f32
    pq_codes_ptr: usize, // [n_vectors * n_subvectors] u8
    inv_list_offsets_ptr: usize, // [n_partitions] u32 — offset into inv_list_ids
    inv_list_lengths_ptr: usize, // [n_partitions] u32
    inv_list_ids_ptr: usize, // [total_ids] u32
    active: bool,
};

var index_slots: [MAX_INDEXES]IndexSlot = [_]IndexSlot{.{
    .dim = 0,
    .n_partitions = 0,
    .n_subvectors = 0,
    .n_codes = 0,
    .n_probe = 0,
    .n_vectors = 0,
    .ivf_centroids_ptr = 0,
    .pq_centroids_ptr = 0,
    .pq_codes_ptr = 0,
    .inv_list_offsets_ptr = 0,
    .inv_list_lengths_ptr = 0,
    .inv_list_ids_ptr = 0,
    .active = false,
}} ** MAX_INDEXES;

// ============================================================================
// WASM Exports
// ============================================================================

/// Load an IVF-PQ index from serialized binary data.
/// Returns handle (1-based slot index), or 0 on failure.
pub export fn ivfPqLoadIndex(data_ptr: [*]const u8, data_len: usize) u32 {
    if (data_len < 28) return 0; // Too small for header

    // Verify magic
    if (data_ptr[0] != 'I' or data_ptr[1] != 'V' or data_ptr[2] != 'P' or data_ptr[3] != 'Q') return 0;

    const data = data_ptr[0..data_len];

    const dim = readU32(data, 4);
    const n_partitions = readU32(data, 8);
    const n_subvectors = readU32(data, 12);
    const n_codes = readU32(data, 16);
    const n_probe = readU32(data, 20);
    const n_vectors = readU32(data, 24);

    if (dim == 0 or n_partitions == 0 or n_subvectors == 0) return 0;
    if (dim % n_subvectors != 0) return 0;

    const subvec_dim = dim / n_subvectors;

    // Calculate expected sizes
    const ivf_centroids_size = n_partitions * dim * 4;
    const pq_centroids_size = n_subvectors * n_codes * subvec_dim * 4;
    const pq_codes_size = n_vectors * n_subvectors;

    var offset: usize = 28;

    // Validate data length (at minimum header + centroids + pq_centroids + codes)
    if (offset + ivf_centroids_size + pq_centroids_size + pq_codes_size > data_len) return 0;

    // Find free slot
    var slot_idx: usize = 0;
    while (slot_idx < MAX_INDEXES) : (slot_idx += 1) {
        if (!index_slots[slot_idx].active) break;
    }
    if (slot_idx >= MAX_INDEXES) return 0;

    // Allocate and copy IVF centroids
    const ivf_ptr = wasmAlloc(ivf_centroids_size) orelse return 0;
    @memcpy(ivf_ptr[0..ivf_centroids_size], data[offset..][0..ivf_centroids_size]);
    offset += ivf_centroids_size;

    // Allocate and copy PQ centroids
    const pq_ptr = wasmAlloc(pq_centroids_size) orelse return 0;
    @memcpy(pq_ptr[0..pq_centroids_size], data[offset..][0..pq_centroids_size]);
    offset += pq_centroids_size;

    // Allocate and copy PQ codes
    const codes_ptr = wasmAlloc(pq_codes_size) orelse return 0;
    @memcpy(codes_ptr[0..pq_codes_size], data[offset..][0..pq_codes_size]);
    offset += pq_codes_size;

    // Parse inverted lists
    // First pass: compute total IDs
    var total_ids: usize = 0;
    var tmp_offset = offset;
    for (0..n_partitions) |_| {
        if (tmp_offset + 4 > data_len) break;
        const list_len = readU32(data, tmp_offset);
        tmp_offset += 4 + list_len * 4;
        total_ids += list_len;
    }

    // Allocate inverted list metadata
    const offsets_ptr = wasmAlloc(n_partitions * 4) orelse return 0;
    const lengths_ptr = wasmAlloc(n_partitions * 4) orelse return 0;
    const ids_ptr = wasmAlloc(if (total_ids > 0) total_ids * 4 else 4) orelse return 0;

    const offsets_u32: [*]u32 = @ptrCast(@alignCast(offsets_ptr));
    const lengths_u32: [*]u32 = @ptrCast(@alignCast(lengths_ptr));
    const ids_u32: [*]u32 = @ptrCast(@alignCast(ids_ptr));

    var id_offset: u32 = 0;
    for (0..n_partitions) |p| {
        if (offset + 4 > data_len) {
            offsets_u32[p] = id_offset;
            lengths_u32[p] = 0;
            continue;
        }
        const list_len = readU32(data, offset);
        offset += 4;

        offsets_u32[p] = id_offset;
        lengths_u32[p] = list_len;

        for (0..list_len) |j| {
            if (offset + 4 <= data_len) {
                ids_u32[id_offset + j] = readU32(data, offset);
                offset += 4;
            }
        }
        id_offset += list_len;
    }

    index_slots[slot_idx] = .{
        .dim = dim,
        .n_partitions = n_partitions,
        .n_subvectors = n_subvectors,
        .n_codes = n_codes,
        .n_probe = n_probe,
        .n_vectors = n_vectors,
        .ivf_centroids_ptr = @intFromPtr(ivf_ptr),
        .pq_centroids_ptr = @intFromPtr(pq_ptr),
        .pq_codes_ptr = @intFromPtr(codes_ptr),
        .inv_list_offsets_ptr = @intFromPtr(offsets_ptr),
        .inv_list_lengths_ptr = @intFromPtr(lengths_ptr),
        .inv_list_ids_ptr = @intFromPtr(ids_ptr),
        .active = true,
    };

    return @intCast(slot_idx + 1); // 1-based handle
}

/// Search the IVF-PQ index for nearest neighbors.
/// Returns number of results written.
pub export fn ivfPqSearch(
    handle: u32,
    query_ptr: [*]const f32,
    dim: u32,
    top_k: u32,
    nprobe: u32,
    out_indices: [*]u32,
    out_scores: [*]f32,
) u32 {
    if (handle == 0 or handle > MAX_INDEXES) return 0;
    const slot = &index_slots[handle - 1];
    if (!slot.active) return 0;
    if (dim != slot.dim) return 0;

    const actual_nprobe = if (nprobe > 0) @min(nprobe, slot.n_partitions) else @min(slot.n_probe, slot.n_partitions);
    const actual_k = @min(top_k, slot.n_vectors);
    if (actual_k == 0) return 0;

    const query = query_ptr[0..dim];
    const subvec_dim = dim / slot.n_subvectors;

    // Get pointers to index data
    const ivf_centroids: [*]const f32 = @ptrFromInt(slot.ivf_centroids_ptr);
    const pq_centroids: [*]const f32 = @ptrFromInt(slot.pq_centroids_ptr);
    const pq_codes: [*]const u8 = @ptrFromInt(slot.pq_codes_ptr);
    const inv_offsets: [*]const u32 = @ptrFromInt(slot.inv_list_offsets_ptr);
    const inv_lengths: [*]const u32 = @ptrFromInt(slot.inv_list_lengths_ptr);
    const inv_ids: [*]const u32 = @ptrFromInt(slot.inv_list_ids_ptr);

    // Find nearest partitions (simple linear scan, store in stack buffer)
    var partition_dists: [256]struct { idx: u32, dist: f32 } = undefined;
    const n_parts = @min(slot.n_partitions, 256);

    for (0..n_parts) |p| {
        var dist: f32 = 0.0;
        for (0..dim) |d| {
            const diff = query[d] - ivf_centroids[p * dim + d];
            dist += diff * diff;
        }
        partition_dists[p] = .{ .idx = @intCast(p), .dist = dist };
    }

    // Partial sort: find top nprobe partitions
    for (0..actual_nprobe) |i| {
        var min_idx = i;
        for (i + 1..n_parts) |j| {
            if (partition_dists[j].dist < partition_dists[min_idx].dist) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            const tmp = partition_dists[i];
            partition_dists[i] = partition_dists[min_idx];
            partition_dists[min_idx] = tmp;
        }
    }

    // Search in top nprobe partitions, maintain top-K with insertion sort
    var result_count: u32 = 0;

    for (0..actual_nprobe) |pi| {
        const part_idx = partition_dists[pi].idx;
        const list_offset = inv_offsets[part_idx];
        const list_len = inv_lengths[part_idx];

        for (0..list_len) |li| {
            const vec_id = inv_ids[list_offset + li];
            if (vec_id >= slot.n_vectors) continue;

            // Compute asymmetric PQ distance
            var dist: f32 = 0.0;
            const codes = pq_codes[vec_id * slot.n_subvectors ..][0..slot.n_subvectors];

            for (0..slot.n_subvectors) |sv| {
                const start = sv * subvec_dim;
                const code = codes[sv];
                const centroid_offset = (sv * slot.n_codes + code) * subvec_dim;

                for (0..subvec_dim) |d| {
                    const diff = query[start + d] - pq_centroids[centroid_offset + d];
                    dist += diff * diff;
                }
            }

            // Insert into top-K results
            var insert_pos: u32 = result_count;
            while (insert_pos > 0 and dist < out_scores[insert_pos - 1]) {
                insert_pos -= 1;
            }

            if (insert_pos < actual_k) {
                if (result_count < actual_k) {
                    var i = result_count;
                    while (i > insert_pos) : (i -= 1) {
                        out_indices[i] = out_indices[i - 1];
                        out_scores[i] = out_scores[i - 1];
                    }
                    result_count += 1;
                } else {
                    var i = actual_k - 1;
                    while (i > insert_pos) : (i -= 1) {
                        out_indices[i] = out_indices[i - 1];
                        out_scores[i] = out_scores[i - 1];
                    }
                }
                out_indices[insert_pos] = vec_id;
                out_scores[insert_pos] = dist;
            }
        }
    }

    return result_count;
}

/// Free an IVF-PQ index by handle.
pub export fn ivfPqFree(handle: u32) void {
    if (handle == 0 or handle > MAX_INDEXES) return;
    index_slots[handle - 1].active = false;
    // Memory is bump-allocated, freed on resetHeap()
}

// ============================================================================
// Helpers
// ============================================================================

fn readU32(data: []const u8, offset: usize) u32 {
    if (offset + 4 > data.len) return 0;
    return @as(u32, data[offset]) |
        (@as(u32, data[offset + 1]) << 8) |
        (@as(u32, data[offset + 2]) << 16) |
        (@as(u32, data[offset + 3]) << 24);
}
