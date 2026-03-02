//! TinyBERT Cross-Encoder (Native)
//!
//! ms-marco-TinyBERT-L-2-v2 cross-encoder for relevance scoring.
//! Uses heap allocation and native SIMD for inference.

const std = @import("std");
const gguf = @import("gguf.zig");

// ============================================================================
// Constants (ms-marco-TinyBERT-L-2-v2)
// ============================================================================

pub const VOCAB_SIZE: usize = 30522;
pub const MAX_SEQ_LEN: usize = 512;
pub const EMBED_DIM: usize = 128;
pub const NUM_HEADS: usize = 2;
pub const NUM_LAYERS: usize = 2;
pub const MLP_DIM: usize = 512;
pub const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // 64

const Vec4 = @Vector(4, f32);
const SIMD_WIDTH: usize = 4;

// ============================================================================
// TinyBERT Model
// ============================================================================

pub const TinyBERT = struct {
    allocator: std.mem.Allocator,
    model: gguf.Model,
    loaded: bool,

    // Tensor references
    word_emb: ?*const gguf.TensorInfo,
    pos_emb: ?*const gguf.TensorInfo,
    token_type_emb: ?*const gguf.TensorInfo,
    emb_ln_weight: ?*const gguf.TensorInfo,
    emb_ln_bias: ?*const gguf.TensorInfo,

    // Per-layer weights
    layer_q_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_q_bias: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_k_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_k_bias: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_v_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_v_bias: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_out_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_out_bias: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_attn_ln_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_attn_ln_bias: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_ffn_up_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_ffn_up_bias: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_ffn_down_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_ffn_down_bias: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_ffn_ln_weight: [NUM_LAYERS]?*const gguf.TensorInfo,
    layer_ffn_ln_bias: [NUM_LAYERS]?*const gguf.TensorInfo,

    // Classification head
    cls_weight: ?*const gguf.TensorInfo,
    cls_bias: ?*const gguf.TensorInfo,

    // Scratch buffers
    hidden: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    attn_buf: []f32,
    attn_out: []f32,
    mlp_hidden: []f32,
    proj_out: []f32,
    weight_row: []f32,
    ln_buf: []f32,

    // Tokenizer buffers
    tokens: []u32,
    attention_mask: []u32,
    token_types: []u32,

    pub fn init(allocator: std.mem.Allocator) !TinyBERT {
        var self = TinyBERT{
            .allocator = allocator,
            .model = gguf.Model.init(allocator),
            .loaded = false,
            .word_emb = null,
            .pos_emb = null,
            .token_type_emb = null,
            .emb_ln_weight = null,
            .emb_ln_bias = null,
            .layer_q_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_q_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_k_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_k_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_v_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_v_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_out_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_out_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_attn_ln_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_attn_ln_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_ffn_up_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_ffn_up_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_ffn_down_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_ffn_down_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_ffn_ln_weight = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .layer_ffn_ln_bias = [_]?*const gguf.TensorInfo{null} ** NUM_LAYERS,
            .cls_weight = null,
            .cls_bias = null,
            .hidden = undefined,
            .q_buf = undefined,
            .k_buf = undefined,
            .v_buf = undefined,
            .attn_buf = undefined,
            .attn_out = undefined,
            .mlp_hidden = undefined,
            .proj_out = undefined,
            .weight_row = undefined,
            .ln_buf = undefined,
            .tokens = undefined,
            .attention_mask = undefined,
            .token_types = undefined,
        };

        // Allocate scratch buffers
        self.hidden = try allocator.alloc(f32, MAX_SEQ_LEN * EMBED_DIM);
        errdefer allocator.free(self.hidden);

        self.q_buf = try allocator.alloc(f32, MAX_SEQ_LEN * EMBED_DIM);
        errdefer allocator.free(self.q_buf);

        self.k_buf = try allocator.alloc(f32, MAX_SEQ_LEN * EMBED_DIM);
        errdefer allocator.free(self.k_buf);

        self.v_buf = try allocator.alloc(f32, MAX_SEQ_LEN * EMBED_DIM);
        errdefer allocator.free(self.v_buf);

        self.attn_buf = try allocator.alloc(f32, NUM_HEADS * MAX_SEQ_LEN * MAX_SEQ_LEN);
        errdefer allocator.free(self.attn_buf);

        self.attn_out = try allocator.alloc(f32, MAX_SEQ_LEN * EMBED_DIM);
        errdefer allocator.free(self.attn_out);

        self.mlp_hidden = try allocator.alloc(f32, MLP_DIM);
        errdefer allocator.free(self.mlp_hidden);

        self.proj_out = try allocator.alloc(f32, EMBED_DIM);
        errdefer allocator.free(self.proj_out);

        self.weight_row = try allocator.alloc(f32, MLP_DIM);
        errdefer allocator.free(self.weight_row);

        self.ln_buf = try allocator.alloc(f32, EMBED_DIM);
        errdefer allocator.free(self.ln_buf);

        self.tokens = try allocator.alloc(u32, MAX_SEQ_LEN);
        errdefer allocator.free(self.tokens);

        self.attention_mask = try allocator.alloc(u32, MAX_SEQ_LEN);
        errdefer allocator.free(self.attention_mask);

        self.token_types = try allocator.alloc(u32, MAX_SEQ_LEN);
        errdefer allocator.free(self.token_types);

        return self;
    }

    pub fn deinit(self: *TinyBERT) void {
        self.allocator.free(self.hidden);
        self.allocator.free(self.q_buf);
        self.allocator.free(self.k_buf);
        self.allocator.free(self.v_buf);
        self.allocator.free(self.attn_buf);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.mlp_hidden);
        self.allocator.free(self.proj_out);
        self.allocator.free(self.weight_row);
        self.allocator.free(self.ln_buf);
        self.allocator.free(self.tokens);
        self.allocator.free(self.attention_mask);
        self.allocator.free(self.token_types);
        self.model.deinit();
    }

    pub fn loadFromFile(self: *TinyBERT, path: []const u8) !void {
        try self.model.loadFromFile(path);
        try self.resolveWeights();
        self.loaded = true;
    }

    fn resolveWeights(self: *TinyBERT) !void {
        // Embedding weights
        self.word_emb = self.model.findTensor("bert.embeddings.word_embeddings.weight");
        self.pos_emb = self.model.findTensor("bert.embeddings.position_embeddings.weight");
        self.token_type_emb = self.model.findTensor("bert.embeddings.token_type_embeddings.weight");
        self.emb_ln_weight = self.model.findTensor("bert.embeddings.LayerNorm.weight");
        self.emb_ln_bias = self.model.findTensor("bert.embeddings.LayerNorm.bias");

        if (self.word_emb == null) return error.MissingWordEmbeddings;
        if (self.pos_emb == null) return error.MissingPositionEmbeddings;
        if (self.emb_ln_weight == null) return error.MissingEmbeddingLayerNorm;

        // Layer weights
        var name_buf: [128]u8 = undefined;
        for (0..NUM_LAYERS) |layer| {
            self.layer_q_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.query.weight", .{layer}) catch continue,
            );
            self.layer_q_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.query.bias", .{layer}) catch continue,
            );
            self.layer_k_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.key.weight", .{layer}) catch continue,
            );
            self.layer_k_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.key.bias", .{layer}) catch continue,
            );
            self.layer_v_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.value.weight", .{layer}) catch continue,
            );
            self.layer_v_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.value.bias", .{layer}) catch continue,
            );
            self.layer_out_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.dense.weight", .{layer}) catch continue,
            );
            self.layer_out_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.dense.bias", .{layer}) catch continue,
            );
            self.layer_attn_ln_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer}) catch continue,
            );
            self.layer_attn_ln_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer}) catch continue,
            );
            self.layer_ffn_up_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.intermediate.dense.weight", .{layer}) catch continue,
            );
            self.layer_ffn_up_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.intermediate.dense.bias", .{layer}) catch continue,
            );
            self.layer_ffn_down_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.dense.weight", .{layer}) catch continue,
            );
            self.layer_ffn_down_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.dense.bias", .{layer}) catch continue,
            );
            self.layer_ffn_ln_weight[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.LayerNorm.weight", .{layer}) catch continue,
            );
            self.layer_ffn_ln_bias[layer] = self.model.findTensor(
                std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.LayerNorm.bias", .{layer}) catch continue,
            );
        }

        // Classification head
        self.cls_weight = self.model.findTensor("classifier.weight") orelse
            self.model.findTensor("classifier.dense.weight") orelse
            self.model.findTensor("classifier.linear.weight");

        self.cls_bias = self.model.findTensor("classifier.bias") orelse
            self.model.findTensor("classifier.dense.bias") orelse
            self.model.findTensor("classifier.linear.bias");

        if (self.cls_weight == null) return error.MissingClassifierHead;
        if (self.cls_bias == null) return error.MissingClassifierBias;
    }

    /// Score a query-document pair using the cross-encoder.
    /// Returns a relevance score between 0 and 1.
    pub fn crossEncode(self: *TinyBERT, query: []const u8, document: []const u8) !f32 {
        if (!self.loaded) return error.ModelNotLoaded;

        // Tokenize
        const seq_len = self.tokenizePair(query, document);
        if (seq_len == 0) return error.TokenizationFailed;

        // Build embeddings
        self.buildEmbeddings(seq_len);

        // Run encoder
        self.runEncoder(seq_len);

        // Classification
        const cls_w = self.cls_weight orelse return error.MissingClassifierHead;
        const cls_b = self.cls_bias orelse return error.MissingClassifierBias;

        // Extract CLS token embedding (position 0)
        const cls_embedding = self.hidden[0..EMBED_DIM];

        // Apply classification: logit = W @ cls + b
        self.model.readWeightRowF32(cls_w, 0, EMBED_DIM, self.weight_row[0..EMBED_DIM]);
        var logit = self.model.readWeightF32(cls_b, 0);

        var j: usize = 0;
        while (j < EMBED_DIM) : (j += SIMD_WIDTH) {
            const va: Vec4 = cls_embedding[j..][0..SIMD_WIDTH].*;
            const vb: Vec4 = self.weight_row[j..][0..SIMD_WIDTH].*;
            logit += @reduce(.Add, va * vb);
        }

        // Sigmoid
        return 1.0 / (1.0 + @exp(-logit));
    }

    // ========================================================================
    // Tokenizer
    // ========================================================================

    fn tokenizePair(self: *TinyBERT, query: []const u8, doc: []const u8) usize {
        if (self.model.vocab_count == 0) return 0;

        var n_tokens: usize = 0;

        // [CLS] token
        self.tokens[n_tokens] = 101;
        self.attention_mask[n_tokens] = 1;
        self.token_types[n_tokens] = 0;
        n_tokens += 1;

        // Tokenize query (segment 0)
        n_tokens = self.tokenizeText(query, n_tokens, 0, MAX_SEQ_LEN - 2);

        // [SEP] after query
        self.tokens[n_tokens] = 102;
        self.attention_mask[n_tokens] = 1;
        self.token_types[n_tokens] = 0;
        n_tokens += 1;

        // Tokenize document (segment 1)
        n_tokens = self.tokenizeText(doc, n_tokens, 1, MAX_SEQ_LEN - 1);

        // [SEP] after document
        self.tokens[n_tokens] = 102;
        self.attention_mask[n_tokens] = 1;
        self.token_types[n_tokens] = 1;
        n_tokens += 1;

        // Pad
        const final_len = n_tokens;
        while (n_tokens < MAX_SEQ_LEN) {
            self.tokens[n_tokens] = 0;
            self.attention_mask[n_tokens] = 0;
            self.token_types[n_tokens] = 0;
            n_tokens += 1;
        }

        return final_len;
    }

    fn tokenizeText(self: *TinyBERT, text: []const u8, start_pos: usize, segment: u32, max_pos: usize) usize {
        var n_tokens = start_pos;
        var text_pos: usize = 0;

        while (text_pos < text.len and n_tokens < max_pos) {
            if (text[text_pos] == 0) break;
            if (text[text_pos] == ' ') {
                text_pos += 1;
                continue;
            }

            var word_end = text_pos;
            while (word_end < text.len and text[word_end] != ' ' and text[word_end] != 0) {
                word_end += 1;
            }

            var word_pos = text_pos;
            var is_first = true;

            while (word_pos < word_end and n_tokens < max_pos) {
                var best_len: usize = 0;
                var best_id: u32 = 100; // [UNK]

                for (0..self.model.vocab_count) |i| {
                    const tok = self.model.getVocabToken(i);
                    if (tok.len == 0) continue;

                    var tok_text = tok;
                    var is_subword = false;
                    if (tok.len >= 2 and tok[0] == '#' and tok[1] == '#') {
                        tok_text = tok[2..];
                        is_subword = true;
                    }

                    if (is_subword == is_first) continue;

                    const remaining = word_end - word_pos;
                    if (tok_text.len > remaining) continue;
                    if (tok_text.len <= best_len) continue;

                    var matches = true;
                    for (0..tok_text.len) |j| {
                        var tc = tok_text[j];
                        var xc = text[word_pos + j];
                        if (tc >= 'A' and tc <= 'Z') tc = tc + 32;
                        if (xc >= 'A' and xc <= 'Z') xc = xc + 32;
                        if (tc != xc) {
                            matches = false;
                            break;
                        }
                    }

                    if (matches) {
                        best_len = tok_text.len;
                        best_id = @intCast(i);
                    }
                }

                if (best_len > 0) {
                    self.tokens[n_tokens] = best_id;
                    self.attention_mask[n_tokens] = 1;
                    self.token_types[n_tokens] = segment;
                    n_tokens += 1;
                    word_pos += best_len;
                    is_first = false;
                } else {
                    word_pos += 1;
                }
            }
            text_pos = word_end;
        }

        return n_tokens;
    }

    // ========================================================================
    // Embeddings
    // ========================================================================

    fn buildEmbeddings(self: *TinyBERT, seq_len: usize) void {
        const word_emb = self.word_emb orelse return;
        const pos_emb = self.pos_emb orelse return;
        const token_type_emb = self.token_type_emb orelse return;
        const emb_ln_w = self.emb_ln_weight orelse return;
        const emb_ln_b = self.emb_ln_bias orelse return;

        for (0..seq_len) |pos| {
            const tok_id = self.tokens[pos];
            const segment_id = self.token_types[pos];
            for (0..EMBED_DIM) |i| {
                self.hidden[pos * EMBED_DIM + i] =
                    self.model.readWeightF32(word_emb, tok_id * EMBED_DIM + i) +
                    self.model.readWeightF32(pos_emb, pos * EMBED_DIM + i) +
                    self.model.readWeightF32(token_type_emb, segment_id * EMBED_DIM + i);
            }
        }

        self.layerNorm(self.hidden, seq_len, emb_ln_w, emb_ln_b);
    }

    fn runEncoder(self: *TinyBERT, seq_len: usize) void {
        for (0..NUM_LAYERS) |layer| {
            self.multiHeadAttention(seq_len, layer);
            self.ffnBlock(seq_len, layer);
        }
    }

    // ========================================================================
    // Neural Network Layers
    // ========================================================================

    fn layerNorm(self: *TinyBERT, input: []f32, seq_len: usize, weight: *const gguf.TensorInfo, bias: *const gguf.TensorInfo) void {
        const eps: f32 = 1e-12;

        for (0..seq_len) |pos| {
            const h = input[pos * EMBED_DIM ..][0..EMBED_DIM];

            // Compute mean
            var sum_vec: Vec4 = @splat(0);
            var i: usize = 0;
            while (i < EMBED_DIM) : (i += SIMD_WIDTH) {
                sum_vec += h[i..][0..SIMD_WIDTH].*;
            }
            const mean = @reduce(.Add, sum_vec) / @as(f32, EMBED_DIM);

            // Compute variance
            const mean_vec: Vec4 = @splat(mean);
            var var_vec: Vec4 = @splat(0);
            i = 0;
            while (i < EMBED_DIM) : (i += SIMD_WIDTH) {
                const diff = h[i..][0..SIMD_WIDTH].* - mean_vec;
                var_vec += diff * diff;
            }
            const variance = @reduce(.Add, var_vec) / @as(f32, EMBED_DIM);
            const inv_std = 1.0 / @sqrt(variance + eps);

            // Read weight and bias
            self.model.readWeightRowF32(weight, 0, EMBED_DIM, self.weight_row[0..EMBED_DIM]);
            self.model.readWeightRowF32(bias, 0, EMBED_DIM, self.ln_buf);

            // Apply normalization
            const inv_std_vec: Vec4 = @splat(inv_std);
            i = 0;
            while (i < EMBED_DIM) : (i += SIMD_WIDTH) {
                const x = h[i..][0..SIMD_WIDTH].*;
                const w: Vec4 = self.weight_row[i..][0..SIMD_WIDTH].*;
                const b: Vec4 = self.ln_buf[i..][0..SIMD_WIDTH].*;
                h[i..][0..SIMD_WIDTH].* = (x - mean_vec) * inv_std_vec * w + b;
            }
        }
    }

    fn linear128to128(self: *TinyBERT, input: []const f32, w: *const gguf.TensorInfo, b: *const gguf.TensorInfo, output: []f32) void {
        for (0..EMBED_DIM) |i| {
            self.model.readWeightRowF32(w, i, EMBED_DIM, self.weight_row[0..EMBED_DIM]);
            var sum: f32 = self.model.readWeightF32(b, i);
            var j: usize = 0;
            while (j < EMBED_DIM) : (j += SIMD_WIDTH) {
                const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
                const vb: Vec4 = self.weight_row[j..][0..SIMD_WIDTH].*;
                sum += @reduce(.Add, va * vb);
            }
            output[i] = sum;
        }
    }

    fn linear128to512(self: *TinyBERT, input: []const f32, w: *const gguf.TensorInfo, b: *const gguf.TensorInfo, output: []f32) void {
        for (0..MLP_DIM) |i| {
            self.model.readWeightRowF32(w, i, EMBED_DIM, self.weight_row[0..EMBED_DIM]);
            var sum: f32 = self.model.readWeightF32(b, i);
            var j: usize = 0;
            while (j < EMBED_DIM) : (j += SIMD_WIDTH) {
                const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
                const vb: Vec4 = self.weight_row[j..][0..SIMD_WIDTH].*;
                sum += @reduce(.Add, va * vb);
            }
            output[i] = sum;
        }
    }

    fn linear512to128(self: *TinyBERT, input: []const f32, w: *const gguf.TensorInfo, b: *const gguf.TensorInfo, output: []f32) void {
        for (0..EMBED_DIM) |i| {
            self.model.readWeightRowF32(w, i, MLP_DIM, self.weight_row[0..MLP_DIM]);
            var sum: f32 = self.model.readWeightF32(b, i);
            var j: usize = 0;
            while (j < MLP_DIM) : (j += SIMD_WIDTH) {
                const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
                const vb: Vec4 = self.weight_row[j..][0..SIMD_WIDTH].*;
                sum += @reduce(.Add, va * vb);
            }
            output[i] = sum;
        }
    }

    fn multiHeadAttention(self: *TinyBERT, seq_len: usize, layer: usize) void {
        const q_w = self.layer_q_weight[layer] orelse return;
        const q_b = self.layer_q_bias[layer] orelse return;
        const k_w = self.layer_k_weight[layer] orelse return;
        const k_b = self.layer_k_bias[layer] orelse return;
        const v_w = self.layer_v_weight[layer] orelse return;
        const v_b = self.layer_v_bias[layer] orelse return;
        const out_w = self.layer_out_weight[layer] orelse return;
        const out_b = self.layer_out_bias[layer] orelse return;
        const ln_w = self.layer_attn_ln_weight[layer] orelse return;
        const ln_b = self.layer_attn_ln_bias[layer] orelse return;

        const scale: f32 = 1.0 / @sqrt(@as(f32, HEAD_DIM));

        // Compute Q, K, V
        for (0..seq_len) |pos| {
            const h = self.hidden[pos * EMBED_DIM ..][0..EMBED_DIM];
            self.linear128to128(h, q_w, q_b, self.q_buf[pos * EMBED_DIM ..][0..EMBED_DIM]);
            self.linear128to128(h, k_w, k_b, self.k_buf[pos * EMBED_DIM ..][0..EMBED_DIM]);
            self.linear128to128(h, v_w, v_b, self.v_buf[pos * EMBED_DIM ..][0..EMBED_DIM]);
        }

        // Attention scores and softmax per head
        for (0..NUM_HEADS) |head| {
            const head_offset = head * HEAD_DIM;

            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    var sum: f32 = 0;
                    var d: usize = 0;
                    while (d < HEAD_DIM) : (d += SIMD_WIDTH) {
                        const qi: Vec4 = self.q_buf[i * EMBED_DIM + head_offset + d ..][0..SIMD_WIDTH].*;
                        const kj: Vec4 = self.k_buf[j * EMBED_DIM + head_offset + d ..][0..SIMD_WIDTH].*;
                        sum += @reduce(.Add, qi * kj);
                    }
                    self.attn_buf[head * seq_len * seq_len + i * seq_len + j] = sum * scale;
                }
            }

            // Softmax per row
            for (0..seq_len) |i| {
                const row_start = head * seq_len * seq_len + i * seq_len;

                var max_val: f32 = self.attn_buf[row_start];
                for (1..seq_len) |j| {
                    if (self.attn_buf[row_start + j] > max_val) max_val = self.attn_buf[row_start + j];
                }

                var sum: f32 = 0;
                for (0..seq_len) |j| {
                    self.attn_buf[row_start + j] = @exp(self.attn_buf[row_start + j] - max_val);
                    sum += self.attn_buf[row_start + j];
                }

                const inv_sum = 1.0 / sum;
                for (0..seq_len) |j| {
                    self.attn_buf[row_start + j] *= inv_sum;
                }
            }
        }

        // Compute attention output
        for (0..seq_len) |pos| {
            for (0..EMBED_DIM) |d| {
                self.attn_out[pos * EMBED_DIM + d] = 0;
            }

            for (0..NUM_HEADS) |head| {
                const head_offset = head * HEAD_DIM;

                for (0..seq_len) |j| {
                    const attn_weight = self.attn_buf[head * seq_len * seq_len + pos * seq_len + j];
                    for (0..HEAD_DIM) |d| {
                        self.attn_out[pos * EMBED_DIM + head_offset + d] +=
                            attn_weight * self.v_buf[j * EMBED_DIM + head_offset + d];
                    }
                }
            }

            // Output projection + residual
            self.linear128to128(self.attn_out[pos * EMBED_DIM ..][0..EMBED_DIM], out_w, out_b, self.proj_out);
            for (0..EMBED_DIM) |d| {
                self.hidden[pos * EMBED_DIM + d] += self.proj_out[d];
            }
        }

        self.layerNorm(self.hidden, seq_len, ln_w, ln_b);
    }

    fn ffnBlock(self: *TinyBERT, seq_len: usize, layer: usize) void {
        const up_w = self.layer_ffn_up_weight[layer] orelse return;
        const up_b = self.layer_ffn_up_bias[layer] orelse return;
        const down_w = self.layer_ffn_down_weight[layer] orelse return;
        const down_b = self.layer_ffn_down_bias[layer] orelse return;
        const ln_w = self.layer_ffn_ln_weight[layer] orelse return;
        const ln_b = self.layer_ffn_ln_bias[layer] orelse return;

        for (0..seq_len) |pos| {
            const h = self.hidden[pos * EMBED_DIM ..][0..EMBED_DIM];

            // Up projection
            self.linear128to512(h, up_w, up_b, self.mlp_hidden);

            // GELU activation
            const sqrt2_inv: f32 = 1.0 / @sqrt(2.0);
            var i: usize = 0;
            while (i < MLP_DIM) : (i += SIMD_WIDTH) {
                const x: Vec4 = self.mlp_hidden[i..][0..SIMD_WIDTH].*;
                var erf_val: Vec4 = undefined;
                inline for (0..SIMD_WIDTH) |k| {
                    erf_val[k] = gguf.erf(x[k] * sqrt2_inv);
                }
                const half: Vec4 = @splat(0.5);
                const one: Vec4 = @splat(1.0);
                self.mlp_hidden[i..][0..SIMD_WIDTH].* = x * half * (one + erf_val);
            }

            // Down projection + residual
            self.linear512to128(self.mlp_hidden, down_w, down_b, self.proj_out);
            for (0..EMBED_DIM) |d| {
                self.hidden[pos * EMBED_DIM + d] += self.proj_out[d];
            }
        }

        self.layerNorm(self.hidden, seq_len, ln_w, ln_b);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "tinybert: initialization" {
    var model = try TinyBERT.init(std.testing.allocator);
    defer model.deinit();

    try std.testing.expect(!model.loaded);
}
