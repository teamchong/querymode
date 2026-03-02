//! CLIP Text Encoder (OpenAI ViT-B/32)
//!
//! Provides text embedding using CLIP's text encoder via ONNX runtime.
//! Outputs 512-dimensional L2-normalized embeddings suitable for
//! cross-modal (text-to-image) search.
//!
//! Model: openai/clip-vit-base-patch32
//! Dimensions: 512
//! Max Sequence Length: 77

const std = @import("std");
const Allocator = std.mem.Allocator;
const Session = @import("session.zig").Session;
const onnx = @import("onnx.zig");

pub const Clip = struct {
    allocator: Allocator,
    session: Session,
    vocab: std.StringHashMap(i64),

    pub const EMBEDDING_DIM: usize = 512;
    pub const MAX_SEQ_LENGTH: usize = 77;
    pub const SOT_TOKEN_ID: i64 = 49406; // <|startoftext|>
    pub const EOT_TOKEN_ID: i64 = 49407; // <|endoftext|>

    const Self = @This();

    /// Initialize CLIP model from ONNX file
    pub fn init(allocator: Allocator, model_path: []const u8) !Self {
        var session = try Session.init(allocator, model_path);
        errdefer session.deinit();

        // Initialize basic vocabulary (for simple tokenization)
        // Full CLIP uses BPE tokenization - this is simplified
        const vocab = std.StringHashMap(i64).init(allocator);

        return Self{
            .allocator = allocator,
            .session = session,
            .vocab = vocab,
        };
    }

    /// Initialize CLIP with vocabulary file
    pub fn initWithVocab(allocator: Allocator, model_path: []const u8, vocab_path: []const u8) !Self {
        var session = try Session.init(allocator, model_path);
        errdefer session.deinit();

        var vocab = std.StringHashMap(i64).init(allocator);
        errdefer vocab.deinit();

        // Load vocabulary from file
        const file = try std.fs.cwd().openFile(vocab_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 50 * 1024 * 1024);
        defer allocator.free(content);

        // Parse JSON vocabulary
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
        defer parsed.deinit();

        if (parsed.value == .object) {
            var it = parsed.value.object.iterator();
            while (it.next()) |entry| {
                const token = entry.key_ptr.*;
                const id: i64 = switch (entry.value_ptr.*) {
                    .integer => |i| i,
                    else => continue,
                };
                const token_copy = try allocator.dupe(u8, token);
                try vocab.put(token_copy, id);
            }
        }

        return Self{
            .allocator = allocator,
            .session = session,
            .vocab = vocab,
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.vocab.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.vocab.deinit();
        self.session.deinit();
    }

    /// Encode text to embedding vector (512 dimensions, L2 normalized)
    pub fn encode(self: *Self, text: []const u8) ![EMBEDDING_DIM]f32 {
        // Tokenize using simple whitespace tokenization
        // Real CLIP uses BPE - this is simplified for demonstration
        var input_ids = try self.allocator.alloc(i64, MAX_SEQ_LENGTH);
        defer self.allocator.free(input_ids);
        @memset(input_ids, 0);

        // Add start token
        input_ids[0] = SOT_TOKEN_ID;
        var pos: usize = 1;

        // Tokenize text (simplified)
        const lower = try self.allocator.alloc(u8, text.len);
        defer self.allocator.free(lower);
        for (text, 0..) |c, i| {
            lower[i] = std.ascii.toLower(c);
        }

        // Simple whitespace tokenization
        var word_start: ?usize = null;
        for (lower, 0..) |c, i| {
            if (std.ascii.isAlphanumeric(c)) {
                if (word_start == null) {
                    word_start = i;
                }
            } else {
                if (word_start) |start| {
                    if (pos < MAX_SEQ_LENGTH - 1) {
                        const word = lower[start..i];
                        const token_id = self.vocab.get(word) orelse @as(i64, @intCast(word.len + 1000));
                        input_ids[pos] = token_id;
                        pos += 1;
                    }
                    word_start = null;
                }
            }
        }
        // Handle last word
        if (word_start) |start| {
            if (pos < MAX_SEQ_LENGTH - 1) {
                const word = lower[start..];
                const token_id = self.vocab.get(word) orelse @as(i64, @intCast(word.len + 1000));
                input_ids[pos] = token_id;
                pos += 1;
            }
        }

        // Add end token
        if (pos < MAX_SEQ_LENGTH) {
            input_ids[pos] = EOT_TOKEN_ID;
        }

        // Create tensor
        const shape = [_]i64{ 1, MAX_SEQ_LENGTH };
        const input_tensor = try self.session.createInt64Tensor(input_ids, &shape);
        defer self.session.releaseTensor(input_tensor);

        // Run inference
        const inputs = [_]?*const onnx.OrtValue{input_tensor};
        const outputs = try self.session.run(&inputs);
        defer self.session.releaseOutputs(outputs);

        // Get embedding from output
        if (outputs.len == 0 or outputs[0] == null) {
            return onnx.OrtError.RunFailed;
        }

        const output_data = try self.session.getFloatData(outputs[0].?);

        var embedding: [EMBEDDING_DIM]f32 = undefined;
        if (output_data.len >= EMBEDDING_DIM) {
            @memcpy(&embedding, output_data[0..EMBEDDING_DIM]);
        } else {
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

/// L2 normalize a vector in-place
fn normalizeL2(embedding: []f32) void {
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

// =============================================================================
// Tests
// =============================================================================

test "clip embedding dim" {
    try std.testing.expectEqual(@as(usize, 512), Clip.EMBEDDING_DIM);
    try std.testing.expectEqual(@as(usize, 77), Clip.MAX_SEQ_LENGTH);
}
