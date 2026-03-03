//! AI Module - Native GGUF Model Inference
//!
//! Provides native implementations of AI models for the CLI.
//! Models are loaded from GGUF files and use SIMD for inference.
//!
//! Available models:
//! - TinyBERT: Cross-encoder for re-ranking (ms-marco-TinyBERT-L-2-v2)
//!
//! Usage:
//! ```zig
//! const ai = @import("querymode.ai");
//! var tinybert = try ai.TinyBERT.init(allocator);
//! defer tinybert.deinit();
//! try tinybert.loadFromFile("models/ms-marco-tinybert-l2.gguf");
//! const score = try tinybert.crossEncode("query", "document");
//! ```

pub const gguf = @import("gguf.zig");
pub const TinyBERT = @import("tinybert.zig").TinyBERT;

// Re-export common types
pub const Model = gguf.Model;
pub const TensorInfo = gguf.TensorInfo;

// Constants
pub const tinybert = struct {
    pub const EMBED_DIM = @import("tinybert.zig").EMBED_DIM;
    pub const NUM_LAYERS = @import("tinybert.zig").NUM_LAYERS;
    pub const NUM_HEADS = @import("tinybert.zig").NUM_HEADS;
    pub const MAX_SEQ_LEN = @import("tinybert.zig").MAX_SEQ_LEN;
};

test {
    _ = gguf;
    _ = @import("tinybert.zig");
}
