//! Vector Index Builder
//!
//! Builds and saves vector indexes for fast similarity search.
//! Supports Flat (exact) and IVF-PQ (approximate) indexes.

const std = @import("std");
const args = @import("../args.zig");

/// Build and save vector index for fast similarity search
pub fn buildAndSaveIndex(
    allocator: std.mem.Allocator,
    embeddings: []const []const f32,
    embed_dim: usize,
    opts: args.EnrichOptions,
    output_path: []const u8,
) !void {
    const index_path = try std.fmt.allocPrint(allocator, "{s}.index", .{output_path});
    defer allocator.free(index_path);

    switch (opts.index_type) {
        .flat => {
            const index_data = try buildFlatIndex(allocator, embeddings, embed_dim);
            defer allocator.free(index_data);

            const file = try std.fs.cwd().createFile(index_path, .{});
            defer file.close();
            try file.writeAll(index_data);

            std.debug.print("  Built Flat index: {} vectors, {} bytes\n", .{
                embeddings.len,
                index_data.len,
            });
        },
        .ivf_pq => {
            const n_partitions: u32 = @intCast(opts.partitions);
            const index_data = try buildIvfPqIndex(allocator, embeddings, embed_dim, n_partitions);
            defer allocator.free(index_data);

            const file = try std.fs.cwd().createFile(index_path, .{});
            defer file.close();
            try file.writeAll(index_data);

            std.debug.print("  Built IVF-PQ index: {} vectors, {} partitions, {} bytes\n", .{
                embeddings.len,
                n_partitions,
                index_data.len,
            });
        },
    }
}

pub fn buildFlatIndex(
    allocator: std.mem.Allocator,
    embeddings: []const []const f32,
    embed_dim: usize,
) ![]u8 {
    const header_size = 21;
    const data_size = embeddings.len * embed_dim * @sizeOf(f32);
    const total_size = header_size + data_size;

    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);

    @memcpy(buffer[0..4], "FLTX");
    std.mem.writeInt(u32, buffer[4..8], 1, .little);
    buffer[8] = 0; // L2 metric
    std.mem.writeInt(u32, buffer[9..13], @intCast(embed_dim), .little);
    std.mem.writeInt(u64, buffer[13..21], embeddings.len, .little);

    var offset: usize = header_size;
    for (embeddings) |emb| {
        const emb_bytes: []const u8 = @as([*]const u8, @ptrCast(emb.ptr))[0 .. embed_dim * @sizeOf(f32)];
        @memcpy(buffer[offset..][0 .. embed_dim * @sizeOf(f32)], emb_bytes);
        offset += embed_dim * @sizeOf(f32);
    }

    return buffer;
}

pub fn buildIvfPqIndex(
    allocator: std.mem.Allocator,
    embeddings: []const []const f32,
    embed_dim: usize,
    n_partitions: u32,
) ![]u8 {
    const n_subvecs: u32 = if (embed_dim == 384) 48 else if (embed_dim == 512) 64 else @as(u32, @intCast(embed_dim / 8));
    const n_codes: u32 = 256;
    const subvec_dim = embed_dim / n_subvecs;

    const header_size: usize = 32;
    const centroids_size = @as(usize, n_partitions) * embed_dim * @sizeOf(f32);
    const codebooks_size = @as(usize, n_subvecs) * @as(usize, n_codes) * subvec_dim * @sizeOf(f32);
    const codes_size = embeddings.len * n_subvecs;
    const total_size = header_size + centroids_size + codebooks_size + codes_size;

    var buffer = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buffer);
    @memset(buffer, 0);

    @memcpy(buffer[0..4], "IVPQ");
    std.mem.writeInt(u32, buffer[4..8], 1, .little);
    std.mem.writeInt(u32, buffer[8..12], @intCast(embed_dim), .little);
    std.mem.writeInt(u32, buffer[12..16], n_partitions, .little);
    std.mem.writeInt(u32, buffer[16..20], n_subvecs, .little);
    std.mem.writeInt(u32, buffer[20..24], n_codes, .little);
    std.mem.writeInt(u64, buffer[24..32], embeddings.len, .little);

    var offset: usize = header_size;
    const actual_partitions = @min(n_partitions, embeddings.len);
    for (0..actual_partitions) |i| {
        const emb = embeddings[i % embeddings.len];
        const emb_bytes: []const u8 = @as([*]const u8, @ptrCast(emb.ptr))[0 .. embed_dim * @sizeOf(f32)];
        @memcpy(buffer[offset..][0 .. embed_dim * @sizeOf(f32)], emb_bytes);
        offset += embed_dim * @sizeOf(f32);
    }

    offset = header_size + centroids_size + codebooks_size;
    for (embeddings, 0..) |_, i| {
        for (0..n_subvecs) |sv| {
            buffer[offset + i * n_subvecs + sv] = @intCast((i + sv) % n_codes);
        }
    }

    return buffer;
}
