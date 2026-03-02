//! Embedding Module
//!
//! Provides text embedding functionality using ONNX runtime.
//! Supports MiniLM (384-dim) and CLIP (512-dim) models.

pub const onnx = @import("onnx.zig");
pub const Session = @import("session.zig").Session;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const MiniLM = @import("minilm.zig").MiniLM;
pub const Clip = @import("clip.zig").Clip;

// Vector index types
pub const flat_index = @import("flat_index.zig");
pub const ivf_pq_index = @import("ivf_pq_index.zig");
pub const FlatIndex = flat_index.FlatIndex;
pub const IvfPqIndex = ivf_pq_index.IvfPqIndex;
pub const DistanceMetric = flat_index.DistanceMetric;
pub const SearchResult = flat_index.SearchResult;
pub const IvfPqConfig = ivf_pq_index.IvfPqConfig;

// Re-export common functions
pub const normalizeL2 = @import("minilm.zig").normalizeL2;
pub const cosineSimilarity = @import("minilm.zig").cosineSimilarity;

/// Model types supported for embedding
pub const ModelType = enum {
    minilm,
    clip,

    pub fn embeddingDim(self: ModelType) usize {
        return switch (self) {
            .minilm => MiniLM.EMBEDDING_DIM,
            .clip => Clip.EMBEDDING_DIM,
        };
    }

    pub fn maxSeqLength(self: ModelType) usize {
        return switch (self) {
            .minilm => MiniLM.MAX_SEQ_LENGTH,
            .clip => Clip.MAX_SEQ_LENGTH,
        };
    }
};

/// Check if ONNX runtime is available
pub fn isOnnxAvailable() bool {
    return onnx.isAvailable();
}

/// Get ONNX runtime version
pub fn getOnnxVersion() []const u8 {
    return onnx.getVersionString();
}

// =============================================================================
// Tests
// =============================================================================

test "model dimensions" {
    const std = @import("std");

    try std.testing.expectEqual(@as(usize, 384), ModelType.embeddingDim(.minilm));
    try std.testing.expectEqual(@as(usize, 512), ModelType.embeddingDim(.clip));

    try std.testing.expectEqual(@as(usize, 256), ModelType.maxSeqLength(.minilm));
    try std.testing.expectEqual(@as(usize, 77), ModelType.maxSeqLength(.clip));
}
