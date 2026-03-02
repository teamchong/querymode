//! MiniLM Text Encoder for WASM
//!
//! all-MiniLM-L6-v2 sentence transformer implementation.
//! Encodes text to 384-dimensional L2-normalized embeddings.

const std = @import("std");
const gguf = @import("gguf_utils.zig");
const format = @import("format.zig");

const readU64LE = format.readU64LE;
const readU32LE = format.readU32LE;
const f16ToF32 = gguf.f16ToF32;
const ggufReadString = gguf.ggufReadString;
const ggufSkipValue = gguf.ggufSkipValue;
const strEql = gguf.strEql;
const erf = gguf.erf;

const GGUF_TYPE_STRING = gguf.GGUF_TYPE_STRING;
const GGUF_TYPE_ARRAY = gguf.GGUF_TYPE_ARRAY;
const GGML_TYPE_F32 = gguf.GGML_TYPE_F32;
const GGML_TYPE_F16 = gguf.GGML_TYPE_F16;

// ============================================================================
// SIMD Types
// ============================================================================

const Vec4 = @Vector(4, f32);
const SIMD_WIDTH: usize = 4;

// ============================================================================
// MiniLM Model Constants
// ============================================================================

const MINILM_VOCAB_SIZE: usize = 30522;
const MINILM_MAX_SEQ_LEN: usize = 256;
const MINILM_EMBED_DIM: usize = 384;
const MINILM_NUM_HEADS: usize = 12;
const MINILM_NUM_LAYERS: usize = 6;
const MINILM_MLP_DIM: usize = 1536;
const MINILM_HEAD_DIM: usize = MINILM_EMBED_DIM / MINILM_NUM_HEADS; // 32

// Comptime assertions for SIMD alignment
comptime {
    if (MINILM_EMBED_DIM % SIMD_WIDTH != 0) @compileError("MINILM_EMBED_DIM must be divisible by SIMD_WIDTH");
    if (MINILM_HEAD_DIM % SIMD_WIDTH != 0) @compileError("MINILM_HEAD_DIM must be divisible by SIMD_WIDTH");
    if (MINILM_MLP_DIM % SIMD_WIDTH != 0) @compileError("MINILM_MLP_DIM must be divisible by SIMD_WIDTH");
}

// ============================================================================
// MiniLM State
// ============================================================================

var minilm_initialized: bool = false;
var minilm_model_loaded: bool = false;

pub fn isLoaded() bool {
    return minilm_model_loaded;
}

// Copy text to internal buffer for embedding
pub fn copyTextToBuffer(text: []const u8) void {
    const len = @min(text.len, minilm_text_buffer.len);
    @memcpy(minilm_text_buffer[0..len], text[0..len]);
}

// Copy output embedding to buffer
pub fn copyOutputToBuffer(buf: []f32) void {
    if (buf.len >= 384) {
        @memcpy(buf[0..384], minilm_output_buffer[0..384]);
    }
}


// Buffers for MiniLM
var minilm_text_buffer: [1024]u8 = undefined;
var minilm_output_buffer: [MINILM_EMBED_DIM]f32 = undefined;

// Model weights storage
var minilm_model_buffer: ?[*]u8 = null;
var minilm_model_size: usize = 0;

// GGUF parsing state
const MAX_TENSORS: usize = 256;
var minilm_gguf_data_offset: usize = 0;
var minilm_n_tensors_loaded: usize = 0;
var minilm_tensor_names: [MAX_TENSORS][96]u8 = undefined;
var minilm_tensor_name_lens: [MAX_TENSORS]usize = undefined;
var minilm_tensor_offsets: [MAX_TENSORS]u64 = undefined;
var minilm_tensor_types: [MAX_TENSORS]u32 = undefined;
var minilm_tensor_dims: [MAX_TENSORS][4]u64 = undefined;
var minilm_tensor_n_dims: [MAX_TENSORS]u32 = undefined;

// MiniLM vocab
var minilm_vocab_data: ?[*]const u8 = null;
var minilm_vocab_count: usize = 0;
var minilm_vocab_string_offsets: [MINILM_VOCAB_SIZE + 1]u32 = undefined;

// MiniLM weight tensor indices
var minilm_word_emb_idx: ?usize = null;
var minilm_pos_emb_idx: ?usize = null;
var minilm_token_type_emb_idx: ?usize = null;
var minilm_emb_ln_weight_idx: ?usize = null;
var minilm_emb_ln_bias_idx: ?usize = null;

// Per-layer weight indices (6 layers)
var minilm_layer_q_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_q_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_k_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_k_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_v_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_v_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_out_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_out_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_attn_ln_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_attn_ln_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_ffn_up_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_ffn_up_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_ffn_down_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_ffn_down_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_ffn_ln_weight_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;
var minilm_layer_ffn_ln_bias_idx: [MINILM_NUM_LAYERS]?usize = [_]?usize{null} ** MINILM_NUM_LAYERS;

// Cross-encoder classification head (384→1)
var minilm_cls_weight_idx: ?usize = null;
var minilm_cls_bias_idx: ?usize = null;
var minilm_classification_output: f32 = 0;
var minilm_cross_encoder_enabled: bool = false;

// Scratch buffers for MiniLM inference
var minilm_scratch_hidden: [MINILM_MAX_SEQ_LEN * MINILM_EMBED_DIM]f32 = undefined;
var minilm_scratch_q: [MINILM_MAX_SEQ_LEN * MINILM_EMBED_DIM]f32 = undefined;
var minilm_scratch_k: [MINILM_MAX_SEQ_LEN * MINILM_EMBED_DIM]f32 = undefined;
var minilm_scratch_v: [MINILM_MAX_SEQ_LEN * MINILM_EMBED_DIM]f32 = undefined;
var minilm_scratch_attn: [MINILM_NUM_HEADS * MINILM_MAX_SEQ_LEN * MINILM_MAX_SEQ_LEN]f32 = undefined;
var minilm_scratch_ln: [MINILM_MAX_SEQ_LEN * MINILM_EMBED_DIM]f32 = undefined;
var minilm_weight_row_buf: [MINILM_MLP_DIM]f32 = undefined;

// Static tokenizer buffers (moved from stack to avoid stack overflow)
var minilm_tokens: [MINILM_MAX_SEQ_LEN]u32 = undefined;
var minilm_attention_mask: [MINILM_MAX_SEQ_LEN]u32 = undefined;
var minilm_token_types: [MINILM_MAX_SEQ_LEN]u32 = undefined; // Segment IDs for cross-encoder

// Cross-encoder buffers (query and document inputs)
var minilm_query_buffer: [512]u8 = undefined;
var minilm_doc_buffer: [512]u8 = undefined;

// Static buffers for attention output and FFN (moved from stack to avoid stack overflow)
var minilm_scratch_attn_output: [MINILM_MAX_SEQ_LEN * MINILM_EMBED_DIM]f32 = undefined;
var minilm_scratch_mlp_hidden: [MINILM_MLP_DIM]f32 = undefined;
var minilm_scratch_proj_out: [MINILM_EMBED_DIM]f32 = undefined;

// ============================================================================
// Tensor Access
// ============================================================================

fn minilmFindTensor(name: []const u8) ?usize {
    for (0..minilm_n_tensors_loaded) |i| {
        if (strEql(minilm_tensor_names[i][0..minilm_tensor_name_lens[i]], name)) {
            return i;
        }
    }
    return null;
}

fn minilmReadWeight(idx: usize, i: usize) f32 {
    const model_data = minilm_model_buffer orelse return 0;
    const offset: usize = minilm_gguf_data_offset + @as(usize, @intCast(minilm_tensor_offsets[idx]));

    if (minilm_tensor_types[idx] == GGML_TYPE_F16) {
        const ptr: [*]const u16 = @ptrCast(@alignCast(model_data + offset));
        return f16ToF32(ptr[i]);
    } else {
        const ptr: [*]const f32 = @ptrCast(@alignCast(model_data + offset));
        return ptr[i];
    }
}

fn minilmGetWeightRowF32(idx: usize, row: usize, row_len: usize, buf: []f32) void {
    const model_data = minilm_model_buffer orelse return;
    const offset: usize = minilm_gguf_data_offset + @as(usize, @intCast(minilm_tensor_offsets[idx]));
    const row_offset = row * row_len;

    if (minilm_tensor_types[idx] == GGML_TYPE_F16) {
        const ptr: [*]const u16 = @ptrCast(@alignCast(model_data + offset));
        for (0..row_len) |i| {
            buf[i] = f16ToF32(ptr[row_offset + i]);
        }
    } else {
        const ptr: [*]const f32 = @ptrCast(@alignCast(model_data + offset));
        @memcpy(buf[0..row_len], ptr[row_offset .. row_offset + row_len]);
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

fn minilmGetVocabToken(idx: usize) []const u8 {
    if (minilm_vocab_data == null or idx >= minilm_vocab_count) return "";
    const start = minilm_vocab_string_offsets[idx];
    const vdata = minilm_vocab_data.?;
    // Safety: read length from 8 bytes at vdata[start..start+8]
    const len: usize = @intCast(
        @as(u64, vdata[start]) |
            (@as(u64, vdata[start + 1]) << 8) |
            (@as(u64, vdata[start + 2]) << 16) |
            (@as(u64, vdata[start + 3]) << 24) |
            (@as(u64, vdata[start + 4]) << 32) |
            (@as(u64, vdata[start + 5]) << 40) |
            (@as(u64, vdata[start + 6]) << 48) |
            (@as(u64, vdata[start + 7]) << 56),
    );
    // Sanity check length
    if (len > 256) return "";
    return vdata[start + 8 .. start + 8 + len];
}

fn minilmTokenize(text: []const u8, tokens: []u32, attention_mask: []u32) usize {

    // Check if vocab is loaded
    if (minilm_vocab_data == null) {
        return 0;
    }
    if (minilm_vocab_count == 0) {
        return 0;
    }

    var n_tokens: usize = 0;

    // [CLS] token
    tokens[n_tokens] = 101;
    attention_mask[n_tokens] = 1;
    n_tokens += 1;

    var text_pos: usize = 0;

    while (text_pos < text.len and n_tokens < MINILM_MAX_SEQ_LEN - 1) {
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

        while (word_pos < word_end and n_tokens < MINILM_MAX_SEQ_LEN - 1) {
            var best_len: usize = 0;
            var best_id: u32 = 100; // [UNK]

            for (0..minilm_vocab_count) |i| {
                const tok = minilmGetVocabToken(i);
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
                tokens[n_tokens] = best_id;
                attention_mask[n_tokens] = 1;
                n_tokens += 1;
                word_pos += best_len;
                is_first = false;
            } else {
                word_pos += 1;
            }
        }

        text_pos = word_end;
    }

    // [SEP] token
    tokens[n_tokens] = 102;
    attention_mask[n_tokens] = 1;
    n_tokens += 1;

    const final_len = n_tokens;
    while (n_tokens < MINILM_MAX_SEQ_LEN) {
        tokens[n_tokens] = 0; // [PAD]
        attention_mask[n_tokens] = 0;
        n_tokens += 1;
    }

    return final_len;
}

/// Tokenize a (query, document) pair for cross-encoder.
/// Format: [CLS] query [SEP] document [SEP]
/// Segment IDs: query tokens = 0, document tokens = 1
fn minilmTokenizePair(
    query: []const u8,
    doc: []const u8,
    tokens: []u32,
    attention_mask: []u32,
    token_types: []u32,
) usize {
    if (minilm_vocab_data == null or minilm_vocab_count == 0) return 0;

    var n_tokens: usize = 0;

    // [CLS] token (segment 0)
    tokens[n_tokens] = 101;
    attention_mask[n_tokens] = 1;
    token_types[n_tokens] = 0;
    n_tokens += 1;

    // Tokenize query (segment 0)
    var text_pos: usize = 0;
    while (text_pos < query.len and n_tokens < MINILM_MAX_SEQ_LEN - 2) {
        if (query[text_pos] == 0) break;
        if (query[text_pos] == ' ') {
            text_pos += 1;
            continue;
        }

        var word_end = text_pos;
        while (word_end < query.len and query[word_end] != ' ' and query[word_end] != 0) {
            word_end += 1;
        }

        var word_pos = text_pos;
        var is_first = true;

        while (word_pos < word_end and n_tokens < MINILM_MAX_SEQ_LEN - 2) {
            var best_len: usize = 0;
            var best_id: u32 = 100; // [UNK]

            for (0..minilm_vocab_count) |i| {
                const tok = minilmGetVocabToken(i);
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
                    var xc = query[word_pos + j];
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
                tokens[n_tokens] = best_id;
                attention_mask[n_tokens] = 1;
                token_types[n_tokens] = 0; // Query segment
                n_tokens += 1;
                word_pos += best_len;
                is_first = false;
            } else {
                word_pos += 1;
            }
        }
        text_pos = word_end;
    }

    // [SEP] after query (segment 0)
    tokens[n_tokens] = 102;
    attention_mask[n_tokens] = 1;
    token_types[n_tokens] = 0;
    n_tokens += 1;

    // Tokenize document (segment 1)
    text_pos = 0;
    while (text_pos < doc.len and n_tokens < MINILM_MAX_SEQ_LEN - 1) {
        if (doc[text_pos] == 0) break;
        if (doc[text_pos] == ' ') {
            text_pos += 1;
            continue;
        }

        var word_end = text_pos;
        while (word_end < doc.len and doc[word_end] != ' ' and doc[word_end] != 0) {
            word_end += 1;
        }

        var word_pos = text_pos;
        var is_first = true;

        while (word_pos < word_end and n_tokens < MINILM_MAX_SEQ_LEN - 1) {
            var best_len: usize = 0;
            var best_id: u32 = 100; // [UNK]

            for (0..minilm_vocab_count) |i| {
                const tok = minilmGetVocabToken(i);
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
                    var xc = doc[word_pos + j];
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
                tokens[n_tokens] = best_id;
                attention_mask[n_tokens] = 1;
                token_types[n_tokens] = 1; // Document segment
                n_tokens += 1;
                word_pos += best_len;
                is_first = false;
            } else {
                word_pos += 1;
            }
        }
        text_pos = word_end;
    }

    // [SEP] after document (segment 1)
    tokens[n_tokens] = 102;
    attention_mask[n_tokens] = 1;
    token_types[n_tokens] = 1;
    n_tokens += 1;

    // Pad remaining
    const final_len = n_tokens;
    while (n_tokens < MINILM_MAX_SEQ_LEN) {
        tokens[n_tokens] = 0; // [PAD]
        attention_mask[n_tokens] = 0;
        token_types[n_tokens] = 0;
        n_tokens += 1;
    }

    return final_len;
}

/// Build token embeddings with optional segment ID support.
/// If use_token_types is false, uses segment 0 for all tokens (bi-encoder mode).
/// If use_token_types is true, uses minilm_token_types array (cross-encoder mode).
fn minilmBuildEmbeddings(seq_len: usize, use_token_types: bool) void {
    const word_emb_idx = minilm_word_emb_idx orelse return;
    const pos_emb_idx = minilm_pos_emb_idx orelse return;
    const token_type_emb_idx = minilm_token_type_emb_idx orelse return;
    const emb_ln_w_idx = minilm_emb_ln_weight_idx orelse return;
    const emb_ln_b_idx = minilm_emb_ln_bias_idx orelse return;

    for (0..seq_len) |pos| {
        const tok_id = minilm_tokens[pos];
        const segment_id = if (use_token_types) minilm_token_types[pos] else 0;
        for (0..MINILM_EMBED_DIM) |i| {
            minilm_scratch_hidden[pos * MINILM_EMBED_DIM + i] =
                minilmReadWeight(word_emb_idx, tok_id * MINILM_EMBED_DIM + i) +
                minilmReadWeight(pos_emb_idx, pos * MINILM_EMBED_DIM + i) +
                minilmReadWeight(token_type_emb_idx, segment_id * MINILM_EMBED_DIM + i);
        }
    }

    minilmLayerNorm(&minilm_scratch_hidden, seq_len, emb_ln_w_idx, emb_ln_b_idx);
}

/// Run BERT encoder layers on minilm_scratch_hidden.
fn minilmRunEncoder(seq_len: usize) void {
    for (0..MINILM_NUM_LAYERS) |layer| {
        minilmMultiHeadAttention(&minilm_scratch_hidden, seq_len, layer);
        minilmFFNBlock(&minilm_scratch_hidden, seq_len, layer);
    }
}

/// Apply mean pooling over valid tokens and L2 normalize.
fn minilmMeanPoolNormalize(seq_len: usize) void {
    for (0..MINILM_EMBED_DIM) |d| {
        minilm_output_buffer[d] = 0;
    }

    var valid_tokens: f32 = 0;
    for (0..seq_len) |pos| {
        if (minilm_attention_mask[pos] == 1) {
            for (0..MINILM_EMBED_DIM) |d| {
                minilm_output_buffer[d] += minilm_scratch_hidden[pos * MINILM_EMBED_DIM + d];
            }
            valid_tokens += 1;
        }
    }

    if (valid_tokens > 0) {
        for (0..MINILM_EMBED_DIM) |d| {
            minilm_output_buffer[d] /= valid_tokens;
        }
    }

    var norm_sq: f32 = 0;
    for (minilm_output_buffer) |v| norm_sq += v * v;
    const norm = @sqrt(norm_sq);
    if (norm > 0) {
        for (&minilm_output_buffer) |*v| v.* /= norm;
    }
}

// ============================================================================
// Neural Network Layers
// ============================================================================

fn minilmLinearLayer384to384(
    input: *const [MINILM_EMBED_DIM]f32,
    w_idx: usize,
    b_idx: usize,
    output: *[MINILM_EMBED_DIM]f32,
) void {
    for (0..MINILM_EMBED_DIM) |i| {
        minilmGetWeightRowF32(w_idx, i, MINILM_EMBED_DIM, minilm_weight_row_buf[0..MINILM_EMBED_DIM]);
        var sum: f32 = minilmReadWeight(b_idx, i);
        var j: usize = 0;
        while (j < MINILM_EMBED_DIM) : (j += SIMD_WIDTH) {
            const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
            const vb: Vec4 = minilm_weight_row_buf[j..][0..SIMD_WIDTH].*;
            sum += @reduce(.Add, va * vb);
        }
        output[i] = sum;
    }
}

fn minilmLinearLayer384to1536(
    input: *const [MINILM_EMBED_DIM]f32,
    w_idx: usize,
    b_idx: usize,
    output: *[MINILM_MLP_DIM]f32,
) void {
    for (0..MINILM_MLP_DIM) |i| {
        minilmGetWeightRowF32(w_idx, i, MINILM_EMBED_DIM, minilm_weight_row_buf[0..MINILM_EMBED_DIM]);
        var sum: f32 = minilmReadWeight(b_idx, i);
        var j: usize = 0;
        while (j < MINILM_EMBED_DIM) : (j += SIMD_WIDTH) {
            const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
            const vb: Vec4 = minilm_weight_row_buf[j..][0..SIMD_WIDTH].*;
            sum += @reduce(.Add, va * vb);
        }
        output[i] = sum;
    }
}

fn minilmLinearLayer1536to384(
    input: *const [MINILM_MLP_DIM]f32,
    w_idx: usize,
    b_idx: usize,
    output: *[MINILM_EMBED_DIM]f32,
) void {
    for (0..MINILM_EMBED_DIM) |i| {
        minilmGetWeightRowF32(w_idx, i, MINILM_MLP_DIM, &minilm_weight_row_buf);
        var sum: f32 = minilmReadWeight(b_idx, i);
        var j: usize = 0;
        while (j < MINILM_MLP_DIM) : (j += SIMD_WIDTH) {
            const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
            const vb: Vec4 = minilm_weight_row_buf[j..][0..SIMD_WIDTH].*;
            sum += @reduce(.Add, va * vb);
        }
        output[i] = sum;
    }
}

fn minilmLayerNorm(input: []f32, seq_len: usize, weight_idx: usize, bias_idx: usize) void {
    const eps: f32 = 1e-12; // BERT uses 1e-12

    for (0..seq_len) |pos| {
        var h: *[MINILM_EMBED_DIM]f32 = @ptrCast(input[pos * MINILM_EMBED_DIM ..][0..MINILM_EMBED_DIM]);

        var sum_vec: Vec4 = @splat(0);
        var i: usize = 0;
        while (i < MINILM_EMBED_DIM) : (i += SIMD_WIDTH) {
            sum_vec += h[i..][0..SIMD_WIDTH].*;
        }
        const mean = @reduce(.Add, sum_vec) / @as(f32, MINILM_EMBED_DIM);

        const mean_vec: Vec4 = @splat(mean);
        var var_vec: Vec4 = @splat(0);
        i = 0;
        while (i < MINILM_EMBED_DIM) : (i += SIMD_WIDTH) {
            const diff = h[i..][0..SIMD_WIDTH].* - mean_vec;
            var_vec += diff * diff;
        }
        const variance = @reduce(.Add, var_vec) / @as(f32, MINILM_EMBED_DIM);
        const inv_std = 1.0 / @sqrt(variance + eps);

        // Use static buffers to avoid stack overflow
        minilmGetWeightRowF32(weight_idx, 0, MINILM_EMBED_DIM, &minilm_weight_row_buf);

        // Read bias into scratch_ln buffer temporarily
        var bias_buf: *[MINILM_EMBED_DIM]f32 = @ptrCast(minilm_scratch_ln[0..MINILM_EMBED_DIM]);
        minilmGetWeightRowF32(bias_idx, 0, MINILM_EMBED_DIM, bias_buf);

        const inv_std_vec: Vec4 = @splat(inv_std);
        i = 0;
        while (i < MINILM_EMBED_DIM) : (i += SIMD_WIDTH) {
            const x = h[i..][0..SIMD_WIDTH].*;
            const w: Vec4 = minilm_weight_row_buf[i..][0..SIMD_WIDTH].*;
            const b: Vec4 = bias_buf[i..][0..SIMD_WIDTH].*;
            h[i..][0..SIMD_WIDTH].* = (x - mean_vec) * inv_std_vec * w + b;
        }
    }
}

fn minilmMultiHeadAttention(input: []f32, seq_len: usize, layer: usize) void {
    const q_w_idx = minilm_layer_q_weight_idx[layer] orelse return;
    const q_b_idx = minilm_layer_q_bias_idx[layer] orelse return;
    const k_w_idx = minilm_layer_k_weight_idx[layer] orelse return;
    const k_b_idx = minilm_layer_k_bias_idx[layer] orelse return;
    const v_w_idx = minilm_layer_v_weight_idx[layer] orelse return;
    const v_b_idx = minilm_layer_v_bias_idx[layer] orelse return;
    const out_w_idx = minilm_layer_out_weight_idx[layer] orelse return;
    const out_b_idx = minilm_layer_out_bias_idx[layer] orelse return;
    const ln_w_idx = minilm_layer_attn_ln_weight_idx[layer] orelse return;
    const ln_b_idx = minilm_layer_attn_ln_bias_idx[layer] orelse return;

    const scale: f32 = 1.0 / @sqrt(@as(f32, MINILM_HEAD_DIM));

    for (0..seq_len) |pos| {
        const h: *const [MINILM_EMBED_DIM]f32 = @ptrCast(input[pos * MINILM_EMBED_DIM ..][0..MINILM_EMBED_DIM]);
        const q: *[MINILM_EMBED_DIM]f32 = @ptrCast(minilm_scratch_q[pos * MINILM_EMBED_DIM ..][0..MINILM_EMBED_DIM]);
        const k: *[MINILM_EMBED_DIM]f32 = @ptrCast(minilm_scratch_k[pos * MINILM_EMBED_DIM ..][0..MINILM_EMBED_DIM]);
        const v: *[MINILM_EMBED_DIM]f32 = @ptrCast(minilm_scratch_v[pos * MINILM_EMBED_DIM ..][0..MINILM_EMBED_DIM]);

        minilmLinearLayer384to384(h, q_w_idx, q_b_idx, q);
        minilmLinearLayer384to384(h, k_w_idx, k_b_idx, k);
        minilmLinearLayer384to384(h, v_w_idx, v_b_idx, v);
    }

    for (0..MINILM_NUM_HEADS) |head| {
        const head_offset = head * MINILM_HEAD_DIM;

        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                var sum: f32 = 0;
                var d: usize = 0;
                while (d < MINILM_HEAD_DIM) : (d += SIMD_WIDTH) {
                    const qi: Vec4 = minilm_scratch_q[i * MINILM_EMBED_DIM + head_offset + d ..][0..SIMD_WIDTH].*;
                    const kj: Vec4 = minilm_scratch_k[j * MINILM_EMBED_DIM + head_offset + d ..][0..SIMD_WIDTH].*;
                    sum += @reduce(.Add, qi * kj);
                }
                minilm_scratch_attn[head * seq_len * seq_len + i * seq_len + j] = sum * scale;
            }
        }

        for (0..seq_len) |i| {
            const row_start = head * seq_len * seq_len + i * seq_len;

            var max_val: f32 = minilm_scratch_attn[row_start];
            for (1..seq_len) |j| {
                if (minilm_scratch_attn[row_start + j] > max_val) max_val = minilm_scratch_attn[row_start + j];
            }

            var sum: f32 = 0;
            for (0..seq_len) |j| {
                minilm_scratch_attn[row_start + j] = @exp(minilm_scratch_attn[row_start + j] - max_val);
                sum += minilm_scratch_attn[row_start + j];
            }

            const inv_sum = 1.0 / sum;
            for (0..seq_len) |j| {
                minilm_scratch_attn[row_start + j] *= inv_sum;
            }
        }
    }

    // Use static buffer instead of stack allocation (was ~384KB on stack!)
    for (0..seq_len) |pos| {
        for (0..MINILM_EMBED_DIM) |d| {
            minilm_scratch_attn_output[pos * MINILM_EMBED_DIM + d] = 0;
        }

        for (0..MINILM_NUM_HEADS) |head| {
            const head_offset = head * MINILM_HEAD_DIM;

            for (0..seq_len) |j| {
                const attn_weight = minilm_scratch_attn[head * seq_len * seq_len + pos * seq_len + j];
                for (0..MINILM_HEAD_DIM) |d| {
                    minilm_scratch_attn_output[pos * MINILM_EMBED_DIM + head_offset + d] +=
                        attn_weight * minilm_scratch_v[j * MINILM_EMBED_DIM + head_offset + d];
                }
            }
        }

        const attn_vec: *const [MINILM_EMBED_DIM]f32 = @ptrCast(minilm_scratch_attn_output[pos * MINILM_EMBED_DIM ..][0..MINILM_EMBED_DIM]);
        // Use static buffer for proj_out instead of stack allocation
        minilmLinearLayer384to384(attn_vec, out_w_idx, out_b_idx, &minilm_scratch_proj_out);

        for (0..MINILM_EMBED_DIM) |d| {
            input[pos * MINILM_EMBED_DIM + d] += minilm_scratch_proj_out[d];
        }
    }

    minilmLayerNorm(input, seq_len, ln_w_idx, ln_b_idx);
}

fn minilmFFNBlock(input: []f32, seq_len: usize, layer: usize) void {
    const up_w_idx = minilm_layer_ffn_up_weight_idx[layer] orelse return;
    const up_b_idx = minilm_layer_ffn_up_bias_idx[layer] orelse return;
    const down_w_idx = minilm_layer_ffn_down_weight_idx[layer] orelse return;
    const down_b_idx = minilm_layer_ffn_down_bias_idx[layer] orelse return;
    const ln_w_idx = minilm_layer_ffn_ln_weight_idx[layer] orelse return;
    const ln_b_idx = minilm_layer_ffn_ln_bias_idx[layer] orelse return;

    for (0..seq_len) |pos| {
        const h: *const [MINILM_EMBED_DIM]f32 = @ptrCast(input[pos * MINILM_EMBED_DIM ..][0..MINILM_EMBED_DIM]);

        // Use static buffer instead of stack allocation
        minilmLinearLayer384to1536(h, up_w_idx, up_b_idx, &minilm_scratch_mlp_hidden);

        const sqrt2_inv: f32 = 1.0 / @sqrt(2.0);
        var i: usize = 0;
        while (i < MINILM_MLP_DIM) : (i += SIMD_WIDTH) {
            const x: Vec4 = minilm_scratch_mlp_hidden[i..][0..SIMD_WIDTH].*;
            var erf_val: Vec4 = undefined;
            inline for (0..SIMD_WIDTH) |k| {
                erf_val[k] = erf(x[k] * sqrt2_inv);
            }
            const half: Vec4 = @splat(0.5);
            const one: Vec4 = @splat(1.0);
            minilm_scratch_mlp_hidden[i..][0..SIMD_WIDTH].* = x * half * (one + erf_val);
        }

        // Use static buffer for ffn_out instead of stack allocation
        minilmLinearLayer1536to384(&minilm_scratch_mlp_hidden, down_w_idx, down_b_idx, &minilm_scratch_proj_out);

        for (0..MINILM_EMBED_DIM) |d| {
            input[pos * MINILM_EMBED_DIM + d] += minilm_scratch_proj_out[d];
        }
    }

    minilmLayerNorm(input, seq_len, ln_w_idx, ln_b_idx);
}

// ============================================================================
// WASM Exports
// ============================================================================

pub export fn minilm_init() i32 {
    minilm_initialized = true;
    minilm_model_loaded = false;

    for (&minilm_output_buffer) |*v| {
        v.* = 0;
    }

    return 0;
}

pub export fn minilm_get_text_buffer() [*]u8 {
    return &minilm_text_buffer;
}

pub export fn minilm_get_text_buffer_size() usize {
    return minilm_text_buffer.len;
}

pub export fn minilm_get_output_buffer() [*]f32 {
    return &minilm_output_buffer;
}

pub export fn minilm_get_output_dim() usize {
    return MINILM_EMBED_DIM;
}

pub export fn minilm_alloc_model_buffer(size: usize) usize {
    const page_size: usize = 65536;
    const pages_needed = (size + page_size - 1) / page_size;

    const current_pages = @wasmMemorySize(0);
    const current_size = current_pages * page_size;

    const result = @wasmMemoryGrow(0, pages_needed);
    if (result == @as(usize, @bitCast(@as(isize, -1)))) {
        return 0;
    }

    const ptr: [*]u8 = @ptrFromInt(current_size);
    minilm_model_buffer = ptr;
    minilm_model_size = size;
    return current_size;
}

pub export fn minilm_weights_loaded() i32 {
    return if (minilm_model_loaded) 1 else 0;
}

pub export fn minilm_load_model(size: usize) i32 {
    const model_data = minilm_model_buffer orelse return -1;
    if (size < 128) return -2;

    const data = model_data[0..size];

    if (data[0] != 'G' or data[1] != 'G' or data[2] != 'U' or data[3] != 'F') {
        return -3;
    }

    const version = readU32LE(data, 4);
    if (version < 2 or version > 3) {
        return -4;
    }

    const n_tensors: usize = @intCast(readU64LE(data, 8));
    const n_kv: usize = @intCast(readU64LE(data, 16));

    var pos: usize = 24;

    for (0..n_kv) |_| {
        const key = ggufReadString(data, &pos);
        if (pos + 4 > data.len) return -5;
        const vtype = readU32LE(data, pos);
        pos += 4;

        if (strEql(key, "tokenizer.ggml.tokens")) {
            if (vtype != GGUF_TYPE_ARRAY) {
                ggufSkipValue(data, &pos, vtype);
                continue;
            }
            if (pos + 12 > data.len) return -6;
            const atype = readU32LE(data, pos);
            pos += 4;
            const alen: usize = @intCast(readU64LE(data, pos));
            pos += 8;

            if (atype == GGUF_TYPE_STRING and alen <= MINILM_VOCAB_SIZE) {
                minilm_vocab_data = data.ptr + pos;
                minilm_vocab_count = alen;

                var str_pos: u32 = 0;
                for (0..alen) |i| {
                    minilm_vocab_string_offsets[i] = str_pos;
                    const slen: u32 = @intCast(readU64LE(data, pos));
                    pos += 8;
                    str_pos += slen + 8;
                    pos += slen;
                }
                minilm_vocab_string_offsets[alen] = str_pos;
            } else {
                ggufSkipValue(data, &pos, vtype);
            }
        } else {
            ggufSkipValue(data, &pos, vtype);
        }
    }

    minilm_n_tensors_loaded = @min(n_tensors, MAX_TENSORS);
    for (0..minilm_n_tensors_loaded) |i| {
        const name = ggufReadString(data, &pos);
        const name_len = @min(name.len, 95);
        @memcpy(minilm_tensor_names[i][0..name_len], name[0..name_len]);
        minilm_tensor_name_lens[i] = name_len;

        if (pos + 4 > data.len) return -7;
        const n_dims = readU32LE(data, pos);
        pos += 4;
        minilm_tensor_n_dims[i] = n_dims;

        for (0..@min(n_dims, 4)) |d| {
            if (pos + 8 > data.len) return -8;
            minilm_tensor_dims[i][d] = readU64LE(data, pos);
            pos += 8;
        }

        if (pos + 12 > data.len) return -9;
        minilm_tensor_types[i] = readU32LE(data, pos);
        pos += 4;
        minilm_tensor_offsets[i] = readU64LE(data, pos);
        pos += 8;
    }

    minilm_gguf_data_offset = (pos + 31) & ~@as(usize, 31);

    minilm_word_emb_idx = minilmFindTensor("bert.embeddings.word_embeddings.weight");
    minilm_pos_emb_idx = minilmFindTensor("bert.embeddings.position_embeddings.weight");
    minilm_token_type_emb_idx = minilmFindTensor("bert.embeddings.token_type_embeddings.weight");
    minilm_emb_ln_weight_idx = minilmFindTensor("bert.embeddings.LayerNorm.weight");
    minilm_emb_ln_bias_idx = minilmFindTensor("bert.embeddings.LayerNorm.bias");

    for (0..MINILM_NUM_LAYERS) |layer| {
        var name_buf: [128]u8 = undefined;

        const q_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.query.weight", .{layer}) catch continue;
        minilm_layer_q_weight_idx[layer] = minilmFindTensor(q_w);

        const q_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.query.bias", .{layer}) catch continue;
        minilm_layer_q_bias_idx[layer] = minilmFindTensor(q_b);

        const k_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.key.weight", .{layer}) catch continue;
        minilm_layer_k_weight_idx[layer] = minilmFindTensor(k_w);

        const k_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.key.bias", .{layer}) catch continue;
        minilm_layer_k_bias_idx[layer] = minilmFindTensor(k_b);

        const v_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.value.weight", .{layer}) catch continue;
        minilm_layer_v_weight_idx[layer] = minilmFindTensor(v_w);

        const v_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.value.bias", .{layer}) catch continue;
        minilm_layer_v_bias_idx[layer] = minilmFindTensor(v_b);

        const out_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.dense.weight", .{layer}) catch continue;
        minilm_layer_out_weight_idx[layer] = minilmFindTensor(out_w);

        const out_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.dense.bias", .{layer}) catch continue;
        minilm_layer_out_bias_idx[layer] = minilmFindTensor(out_b);

        const attn_ln_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer}) catch continue;
        minilm_layer_attn_ln_weight_idx[layer] = minilmFindTensor(attn_ln_w);

        const attn_ln_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer}) catch continue;
        minilm_layer_attn_ln_bias_idx[layer] = minilmFindTensor(attn_ln_b);

        const up_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.intermediate.dense.weight", .{layer}) catch continue;
        minilm_layer_ffn_up_weight_idx[layer] = minilmFindTensor(up_w);

        const up_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.intermediate.dense.bias", .{layer}) catch continue;
        minilm_layer_ffn_up_bias_idx[layer] = minilmFindTensor(up_b);

        const down_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.dense.weight", .{layer}) catch continue;
        minilm_layer_ffn_down_weight_idx[layer] = minilmFindTensor(down_w);

        const down_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.dense.bias", .{layer}) catch continue;
        minilm_layer_ffn_down_bias_idx[layer] = minilmFindTensor(down_b);

        const ffn_ln_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.LayerNorm.weight", .{layer}) catch continue;
        minilm_layer_ffn_ln_weight_idx[layer] = minilmFindTensor(ffn_ln_w);

        const ffn_ln_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.LayerNorm.bias", .{layer}) catch continue;
        minilm_layer_ffn_ln_bias_idx[layer] = minilmFindTensor(ffn_ln_b);
    }

    // Cross-encoder classification head (optional - only present in cross-encoder models)
    // Try multiple naming conventions used by different cross-encoder models
    minilm_cls_weight_idx = minilmFindTensor("classifier.dense.weight");
    if (minilm_cls_weight_idx == null) {
        minilm_cls_weight_idx = minilmFindTensor("classifier.weight");
    }
    if (minilm_cls_weight_idx == null) {
        minilm_cls_weight_idx = minilmFindTensor("classifier.linear.weight");
    }

    minilm_cls_bias_idx = minilmFindTensor("classifier.dense.bias");
    if (minilm_cls_bias_idx == null) {
        minilm_cls_bias_idx = minilmFindTensor("classifier.bias");
    }
    if (minilm_cls_bias_idx == null) {
        minilm_cls_bias_idx = minilmFindTensor("classifier.linear.bias");
    }

    // Enable cross-encoder mode if classification head weights were found
    minilm_cross_encoder_enabled = (minilm_cls_weight_idx != null);

    if (minilm_word_emb_idx == null) return -10;
    if (minilm_pos_emb_idx == null) return -11;
    if (minilm_emb_ln_weight_idx == null) return -12;

    minilm_model_loaded = true;
    return 0;
}

pub export fn minilm_encode_text(text_len: usize) i32 {
    if (!minilm_initialized) return -1;
    if (!minilm_model_loaded) return -2;
    if (text_len == 0 or text_len > minilm_text_buffer.len) return -3;

    if (minilm_word_emb_idx == null) return -4;
    if (minilm_pos_emb_idx == null) return -5;
    if (minilm_token_type_emb_idx == null) return -6;
    if (minilm_emb_ln_weight_idx == null) return -7;
    if (minilm_emb_ln_bias_idx == null) return -8;

    // Tokenize and encode using helper functions
    const seq_len = minilmTokenize(minilm_text_buffer[0..text_len], &minilm_tokens, &minilm_attention_mask);
    minilmBuildEmbeddings(seq_len, false); // false = bi-encoder mode (segment 0 for all)
    minilmRunEncoder(seq_len);
    minilmMeanPoolNormalize(seq_len);

    return 0;
}

// ============================================================================
// Cross-Encoder Exports
// ============================================================================

/// Check if cross-encoder mode is available (classification head loaded).
pub export fn minilm_is_cross_encoder() i32 {
    return if (minilm_cross_encoder_enabled) 1 else 0;
}

/// Get pointer to query input buffer for cross-encoder.
pub export fn minilm_get_query_buffer() [*]u8 {
    return &minilm_query_buffer;
}

/// Get size of query buffer.
pub export fn minilm_get_query_buffer_size() usize {
    return minilm_query_buffer.len;
}

/// Get pointer to document input buffer for cross-encoder.
pub export fn minilm_get_doc_buffer() [*]u8 {
    return &minilm_doc_buffer;
}

/// Get size of document buffer.
pub export fn minilm_get_doc_buffer_size() usize {
    return minilm_doc_buffer.len;
}

/// Get the classification score from the last cross_encode call.
pub export fn minilm_get_score() f32 {
    return minilm_classification_output;
}

/// Score a (query, document) pair using cross-encoder.
/// Returns 0 on success, negative error code on failure.
/// Result stored in minilm_classification_output (use minilm_get_score() to retrieve).
pub export fn minilm_cross_encode(query_len: usize, doc_len: usize) i32 {
    if (!minilm_initialized) return -1;
    if (!minilm_model_loaded) return -2;
    if (!minilm_cross_encoder_enabled) return -3; // No classification head
    if (query_len == 0 or query_len > minilm_query_buffer.len) return -4;
    if (doc_len == 0 or doc_len > minilm_doc_buffer.len) return -5;

    if (minilm_word_emb_idx == null) return -6;
    if (minilm_pos_emb_idx == null) return -7;
    if (minilm_token_type_emb_idx == null) return -8;
    if (minilm_emb_ln_weight_idx == null) return -9;
    if (minilm_emb_ln_bias_idx == null) return -10;

    const cls_w_idx = minilm_cls_weight_idx orelse return -11;
    const cls_b_idx = minilm_cls_bias_idx orelse return -12;

    // Tokenize query-document pair with segment IDs
    const seq_len = minilmTokenizePair(
        minilm_query_buffer[0..query_len],
        minilm_doc_buffer[0..doc_len],
        &minilm_tokens,
        &minilm_attention_mask,
        &minilm_token_types,
    );

    if (seq_len == 0) return -13;

    // Build embeddings with token types (cross-encoder mode)
    minilmBuildEmbeddings(seq_len, true);

    // Run BERT encoder
    minilmRunEncoder(seq_len);

    // Extract CLS token embedding (position 0)
    const cls_embedding: *const [MINILM_EMBED_DIM]f32 = @ptrCast(minilm_scratch_hidden[0..MINILM_EMBED_DIM]);

    // Apply classification head: score = sigmoid(W @ cls + b)
    // W is (1, 384) or (384,) and b is (1,) or scalar

    // Read classification weights into buffer once
    minilmGetWeightRowF32(cls_w_idx, 0, MINILM_EMBED_DIM, minilm_weight_row_buf[0..MINILM_EMBED_DIM]);

    // Start with bias
    var logit: f32 = minilmReadWeight(cls_b_idx, 0);

    // Compute dot product: logit += W @ cls
    var j: usize = 0;
    while (j < MINILM_EMBED_DIM) : (j += SIMD_WIDTH) {
        const va: Vec4 = cls_embedding[j..][0..SIMD_WIDTH].*;
        const vb: Vec4 = minilm_weight_row_buf[j..][0..SIMD_WIDTH].*;
        logit += @reduce(.Add, va * vb);
    }

    // Apply sigmoid: score = 1 / (1 + exp(-logit))
    minilm_classification_output = 1.0 / (1.0 + @exp(-logit));

    return 0;
}

// ============================================================================
// Tests
// ============================================================================

test "minilm_model: initialization" {
    minilm_initialized = false;
    minilm_model_loaded = false;
    const result = minilm_init();
    try std.testing.expectEqual(@as(i32, 0), result);
    try std.testing.expect(minilm_initialized);
    try std.testing.expect(!minilm_model_loaded);
}

test "minilm_model: buffer access" {
    const text_buf = minilm_get_text_buffer();
    try std.testing.expect(text_buf != undefined);

    const text_size = minilm_get_text_buffer_size();
    try std.testing.expectEqual(@as(usize, 1024), text_size);

    const out_buf = minilm_get_output_buffer();
    try std.testing.expect(out_buf != undefined);

    const out_dim = minilm_get_output_dim();
    try std.testing.expectEqual(@as(usize, 384), out_dim);
}

test "minilm_model: weights not loaded" {
    minilm_initialized = true;
    minilm_model_loaded = false;
    const result = minilm_weights_loaded();
    try std.testing.expectEqual(@as(i32, 0), result);
}
