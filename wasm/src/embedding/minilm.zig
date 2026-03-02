//! MiniLM Embedding Model (all-MiniLM-L6-v2)
//!
//! Provides text embedding using the all-MiniLM-L6-v2 model via ONNX runtime.
//! Outputs 384-dimensional L2-normalized embeddings.
//!
//! Model: sentence-transformers/all-MiniLM-L6-v2
//! Dimensions: 384
//! Max Sequence Length: 256

const std = @import("std");
const Allocator = std.mem.Allocator;
const Session = @import("session.zig").Session;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const onnx = @import("onnx.zig");

pub const MiniLM = struct {
    allocator: Allocator,
    session: Session,
    tokenizer: Tokenizer,

    pub const EMBEDDING_DIM: usize = 384;
    pub const MAX_SEQ_LENGTH: usize = 256;

    const Self = @This();

    /// Initialize MiniLM model from ONNX file and vocabulary
    pub fn init(allocator: Allocator, model_path: []const u8, vocab_path: []const u8) !Self {
        var session = try Session.init(allocator, model_path);
        errdefer session.deinit();

        var tokenizer = try Tokenizer.initFromFile(allocator, vocab_path);
        errdefer tokenizer.deinit();

        return Self{
            .allocator = allocator,
            .session = session,
            .tokenizer = tokenizer,
        };
    }

    /// Initialize with basic tokenizer (for testing without vocab file)
    pub fn initWithBasicTokenizer(allocator: Allocator, model_path: []const u8) !Self {
        var session = try Session.init(allocator, model_path);
        errdefer session.deinit();

        var tokenizer = try Tokenizer.initBasic(allocator);
        errdefer tokenizer.deinit();

        return Self{
            .allocator = allocator,
            .session = session,
            .tokenizer = tokenizer,
        };
    }

    pub fn deinit(self: *Self) void {
        self.tokenizer.deinit();
        self.session.deinit();
    }

    /// Encode text to embedding vector (384 dimensions, L2 normalized)
    pub fn encode(self: *Self, text: []const u8) ![EMBEDDING_DIM]f32 {
        // Tokenize
        const encoded = try self.tokenizer.encode(text, MAX_SEQ_LENGTH);
        defer self.tokenizer.freeEncoded(encoded);

        // Create tensors
        const shape = [_]i64{ 1, MAX_SEQ_LENGTH };

        const input_ids_tensor = try self.session.createInt64Tensor(encoded.input_ids, &shape);
        defer self.session.releaseTensor(input_ids_tensor);

        const attention_mask_tensor = try self.session.createInt64Tensor(encoded.attention_mask, &shape);
        defer self.session.releaseTensor(attention_mask_tensor);

        // Token type IDs (all zeros for single sentence)
        const token_type_ids = try self.allocator.alloc(i64, MAX_SEQ_LENGTH);
        defer self.allocator.free(token_type_ids);
        @memset(token_type_ids, 0);

        const token_type_tensor = try self.session.createInt64Tensor(token_type_ids, &shape);
        defer self.session.releaseTensor(token_type_tensor);

        // Run inference
        const inputs = [_]?*const onnx.OrtValue{
            input_ids_tensor,
            attention_mask_tensor,
            token_type_tensor,
        };

        const outputs = try self.session.run(&inputs);
        defer self.session.releaseOutputs(outputs);

        // Get embedding from output
        // MiniLM outputs: last_hidden_state [1, seq_len, 384] or sentence_embedding [1, 384]
        if (outputs.len == 0 or outputs[0] == null) {
            return onnx.OrtError.RunFailed;
        }

        const output_data = try self.session.getFloatData(outputs[0].?);

        // Mean pooling over sequence length if needed
        var embedding: [EMBEDDING_DIM]f32 = undefined;

        if (output_data.len == EMBEDDING_DIM) {
            // Direct sentence embedding output
            @memcpy(&embedding, output_data[0..EMBEDDING_DIM]);
        } else if (output_data.len >= MAX_SEQ_LENGTH * EMBEDDING_DIM) {
            // Last hidden state - need mean pooling
            embedding = meanPool(output_data, encoded.attention_mask, MAX_SEQ_LENGTH, EMBEDDING_DIM);
        } else {
            // Unexpected output shape
            return onnx.OrtError.GetTensorDataFailed;
        }

        // L2 normalize
        normalizeL2(&embedding);

        return embedding;
    }

    /// Encode multiple texts to embeddings (batch processing)
    pub fn encodeBatch(self: *Self, texts: []const []const u8) ![]const [EMBEDDING_DIM]f32 {
        var embeddings = try self.allocator.alloc([EMBEDDING_DIM]f32, texts.len);
        errdefer self.allocator.free(embeddings);

        for (texts, 0..) |text, i| {
            embeddings[i] = try self.encode(text);
        }

        return embeddings;
    }

    /// Free batch embeddings
    pub fn freeBatch(self: *Self, embeddings: []const [EMBEDDING_DIM]f32) void {
        self.allocator.free(embeddings);
    }
};

/// Mean pooling over sequence dimension with attention mask
fn meanPool(
    hidden_states: []const f32,
    attention_mask: []const i64,
    seq_len: usize,
    hidden_size: usize,
) [MiniLM.EMBEDDING_DIM]f32 {
    var result: [MiniLM.EMBEDDING_DIM]f32 = [_]f32{0.0} ** MiniLM.EMBEDDING_DIM;
    var total_weight: f32 = 0.0;

    for (0..seq_len) |i| {
        const mask_val: f32 = @floatFromInt(attention_mask[i]);
        if (mask_val > 0) {
            total_weight += mask_val;
            for (0..hidden_size) |j| {
                result[j] += hidden_states[i * hidden_size + j] * mask_val;
            }
        }
    }

    // Divide by total weight
    if (total_weight > 0) {
        for (0..hidden_size) |j| {
            result[j] /= total_weight;
        }
    }

    return result;
}

/// L2 normalize a vector in-place
pub fn normalizeL2(embedding: []f32) void {
    var norm: f32 = 0.0;
    for (embedding) |val| {
        norm += val * val;
    }
    norm = @sqrt(norm);

    if (norm > 0) {
        for (embedding) |*val| {
            val.* /= norm;
        }
    }
}

/// Compute cosine similarity between two embeddings
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0.0;

    var dot: f32 = 0.0;
    for (a, b) |va, vb| {
        dot += va * vb;
    }

    // If vectors are L2-normalized, dot product equals cosine similarity
    return dot;
}

// =============================================================================
// Tests
// =============================================================================

test "normalize L2" {
    var vec = [_]f32{ 3.0, 4.0 };
    normalizeL2(&vec);

    // 3/5 = 0.6, 4/5 = 0.8
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), vec[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vec[1], 0.001);
}

test "cosine similarity" {
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0 };
    const c = [_]f32{ 1.0, 0.0 };

    // Orthogonal vectors
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &b), 0.001);

    // Identical vectors (normalized)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &c), 0.001);
}
