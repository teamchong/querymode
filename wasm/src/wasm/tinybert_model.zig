//! TinyBERT Cross-Encoder for WASM
//!
//! ms-marco-TinyBERT-L-2-v2 cross-encoder implementation.
//! Scores (query, document) pairs for re-ranking.
//! Optimized for small memory footprint (~4.5MB model).

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
// TinyBERT Model Constants (ms-marco-TinyBERT-L-2-v2)
// ============================================================================

const TINY_VOCAB_SIZE: usize = 30522;
const TINY_MAX_SEQ_LEN: usize = 512;
const TINY_EMBED_DIM: usize = 128;
const TINY_NUM_HEADS: usize = 2;
const TINY_NUM_LAYERS: usize = 2;
const TINY_MLP_DIM: usize = 512;
const TINY_HEAD_DIM: usize = TINY_EMBED_DIM / TINY_NUM_HEADS; // 64

// Comptime assertions for SIMD alignment
comptime {
    if (TINY_EMBED_DIM % SIMD_WIDTH != 0) @compileError("TINY_EMBED_DIM must be divisible by SIMD_WIDTH");
    if (TINY_HEAD_DIM % SIMD_WIDTH != 0) @compileError("TINY_HEAD_DIM must be divisible by SIMD_WIDTH");
    if (TINY_MLP_DIM % SIMD_WIDTH != 0) @compileError("TINY_MLP_DIM must be divisible by SIMD_WIDTH");
}

// ============================================================================
// TinyBERT State
// ============================================================================

var tiny_initialized: bool = false;
var tiny_model_loaded: bool = false;

pub fn isLoaded() bool {
    return tiny_model_loaded;
}

// Input buffers for cross-encoder (query + document)
var tiny_query_buffer: [512]u8 = undefined;
var tiny_doc_buffer: [512]u8 = undefined;

// Classification output
var tiny_classification_output: f32 = 0;

// Model weights storage
var tiny_model_buffer: ?[*]u8 = null;
var tiny_model_size: usize = 0;

// GGUF parsing state
const MAX_TENSORS: usize = 128; // TinyBERT has fewer tensors
var tiny_gguf_data_offset: usize = 0;
var tiny_n_tensors_loaded: usize = 0;
var tiny_tensor_names: [MAX_TENSORS][96]u8 = undefined;
var tiny_tensor_name_lens: [MAX_TENSORS]usize = undefined;
var tiny_tensor_offsets: [MAX_TENSORS]u64 = undefined;
var tiny_tensor_types: [MAX_TENSORS]u32 = undefined;
var tiny_tensor_dims: [MAX_TENSORS][4]u64 = undefined;
var tiny_tensor_n_dims: [MAX_TENSORS]u32 = undefined;

// TinyBERT vocab
var tiny_vocab_data: ?[*]const u8 = null;
var tiny_vocab_count: usize = 0;
var tiny_vocab_string_offsets: [TINY_VOCAB_SIZE + 1]u32 = undefined;

// TinyBERT weight tensor indices
var tiny_word_emb_idx: ?usize = null;
var tiny_pos_emb_idx: ?usize = null;
var tiny_token_type_emb_idx: ?usize = null;
var tiny_emb_ln_weight_idx: ?usize = null;
var tiny_emb_ln_bias_idx: ?usize = null;

// Per-layer weight indices (2 layers)
var tiny_layer_q_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_q_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_k_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_k_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_v_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_v_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_out_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_out_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_attn_ln_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_attn_ln_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_ffn_up_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_ffn_up_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_ffn_down_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_ffn_down_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_ffn_ln_weight_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;
var tiny_layer_ffn_ln_bias_idx: [TINY_NUM_LAYERS]?usize = [_]?usize{null} ** TINY_NUM_LAYERS;

// Classification head (cross-encoder specific)
var tiny_cls_weight_idx: ?usize = null;
var tiny_cls_bias_idx: ?usize = null;

// Scratch buffers for TinyBERT inference (much smaller than MiniLM!)
// 512 * 128 * 4 = 256KB per buffer
var tiny_scratch_hidden: [TINY_MAX_SEQ_LEN * TINY_EMBED_DIM]f32 = undefined;
var tiny_scratch_q: [TINY_MAX_SEQ_LEN * TINY_EMBED_DIM]f32 = undefined;
var tiny_scratch_k: [TINY_MAX_SEQ_LEN * TINY_EMBED_DIM]f32 = undefined;
var tiny_scratch_v: [TINY_MAX_SEQ_LEN * TINY_EMBED_DIM]f32 = undefined;
var tiny_scratch_attn: [TINY_NUM_HEADS * TINY_MAX_SEQ_LEN * TINY_MAX_SEQ_LEN]f32 = undefined;
var tiny_scratch_ln: [TINY_MAX_SEQ_LEN * TINY_EMBED_DIM]f32 = undefined;
var tiny_weight_row_buf: [TINY_MLP_DIM]f32 = undefined;

// Static tokenizer buffers
var tiny_tokens: [TINY_MAX_SEQ_LEN]u32 = undefined;
var tiny_attention_mask: [TINY_MAX_SEQ_LEN]u32 = undefined;
var tiny_token_types: [TINY_MAX_SEQ_LEN]u32 = undefined;

// Static buffers for attention output and FFN
var tiny_scratch_attn_output: [TINY_MAX_SEQ_LEN * TINY_EMBED_DIM]f32 = undefined;
var tiny_scratch_mlp_hidden: [TINY_MLP_DIM]f32 = undefined;
var tiny_scratch_proj_out: [TINY_EMBED_DIM]f32 = undefined;

// ============================================================================
// Tensor Access
// ============================================================================

fn tinyFindTensor(name: []const u8) ?usize {
    for (0..tiny_n_tensors_loaded) |i| {
        if (strEql(tiny_tensor_names[i][0..tiny_tensor_name_lens[i]], name)) {
            return i;
        }
    }
    return null;
}

fn tinyReadWeight(idx: usize, i: usize) f32 {
    const model_data = tiny_model_buffer orelse return 0;
    const offset: usize = tiny_gguf_data_offset + @as(usize, @intCast(tiny_tensor_offsets[idx]));

    if (tiny_tensor_types[idx] == GGML_TYPE_F16) {
        const ptr: [*]const u16 = @ptrCast(@alignCast(model_data + offset));
        return f16ToF32(ptr[i]);
    } else {
        const ptr: [*]const f32 = @ptrCast(@alignCast(model_data + offset));
        return ptr[i];
    }
}

fn tinyGetWeightRowF32(idx: usize, row: usize, row_len: usize, buf: []f32) void {
    const model_data = tiny_model_buffer orelse return;
    const offset: usize = tiny_gguf_data_offset + @as(usize, @intCast(tiny_tensor_offsets[idx]));
    const row_offset = row * row_len;

    if (tiny_tensor_types[idx] == GGML_TYPE_F16) {
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

fn tinyGetVocabToken(idx: usize) []const u8 {
    if (tiny_vocab_data == null or idx >= tiny_vocab_count) return "";
    const start = tiny_vocab_string_offsets[idx];
    const vdata = tiny_vocab_data.?;
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
    if (len > 256) return "";
    return vdata[start + 8 .. start + 8 + len];
}

/// Tokenize a (query, document) pair for cross-encoder.
/// Format: [CLS] query [SEP] document [SEP]
/// Segment IDs: query tokens = 0, document tokens = 1
fn tinyTokenizePair(
    query: []const u8,
    doc: []const u8,
    tokens: []u32,
    attention_mask: []u32,
    token_types: []u32,
) usize {
    if (tiny_vocab_data == null or tiny_vocab_count == 0) return 0;

    var n_tokens: usize = 0;

    // [CLS] token (segment 0)
    tokens[n_tokens] = 101;
    attention_mask[n_tokens] = 1;
    token_types[n_tokens] = 0;
    n_tokens += 1;

    // Tokenize query (segment 0)
    var text_pos: usize = 0;
    while (text_pos < query.len and n_tokens < TINY_MAX_SEQ_LEN - 2) {
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

        while (word_pos < word_end and n_tokens < TINY_MAX_SEQ_LEN - 2) {
            var best_len: usize = 0;
            var best_id: u32 = 100; // [UNK]

            for (0..tiny_vocab_count) |i| {
                const tok = tinyGetVocabToken(i);
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
                token_types[n_tokens] = 0;
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
    while (text_pos < doc.len and n_tokens < TINY_MAX_SEQ_LEN - 1) {
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

        while (word_pos < word_end and n_tokens < TINY_MAX_SEQ_LEN - 1) {
            var best_len: usize = 0;
            var best_id: u32 = 100; // [UNK]

            for (0..tiny_vocab_count) |i| {
                const tok = tinyGetVocabToken(i);
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
                token_types[n_tokens] = 1;
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
    while (n_tokens < TINY_MAX_SEQ_LEN) {
        tokens[n_tokens] = 0;
        attention_mask[n_tokens] = 0;
        token_types[n_tokens] = 0;
        n_tokens += 1;
    }

    return final_len;
}

// ============================================================================
// Embedding and Encoder Helpers
// ============================================================================

fn tinyBuildEmbeddings(seq_len: usize) void {
    const word_emb_idx = tiny_word_emb_idx orelse return;
    const pos_emb_idx = tiny_pos_emb_idx orelse return;
    const token_type_emb_idx = tiny_token_type_emb_idx orelse return;
    const emb_ln_w_idx = tiny_emb_ln_weight_idx orelse return;
    const emb_ln_b_idx = tiny_emb_ln_bias_idx orelse return;

    for (0..seq_len) |pos| {
        const tok_id = tiny_tokens[pos];
        const segment_id = tiny_token_types[pos];
        for (0..TINY_EMBED_DIM) |i| {
            tiny_scratch_hidden[pos * TINY_EMBED_DIM + i] =
                tinyReadWeight(word_emb_idx, tok_id * TINY_EMBED_DIM + i) +
                tinyReadWeight(pos_emb_idx, pos * TINY_EMBED_DIM + i) +
                tinyReadWeight(token_type_emb_idx, segment_id * TINY_EMBED_DIM + i);
        }
    }

    tinyLayerNorm(&tiny_scratch_hidden, seq_len, emb_ln_w_idx, emb_ln_b_idx);
}

fn tinyRunEncoder(seq_len: usize) void {
    for (0..TINY_NUM_LAYERS) |layer| {
        tinyMultiHeadAttention(&tiny_scratch_hidden, seq_len, layer);
        tinyFFNBlock(&tiny_scratch_hidden, seq_len, layer);
    }
}

// ============================================================================
// Neural Network Layers
// ============================================================================

fn tinyLinearLayer128to128(
    input: *const [TINY_EMBED_DIM]f32,
    w_idx: usize,
    b_idx: usize,
    output: *[TINY_EMBED_DIM]f32,
) void {
    for (0..TINY_EMBED_DIM) |i| {
        tinyGetWeightRowF32(w_idx, i, TINY_EMBED_DIM, tiny_weight_row_buf[0..TINY_EMBED_DIM]);
        var sum: f32 = tinyReadWeight(b_idx, i);
        var j: usize = 0;
        while (j < TINY_EMBED_DIM) : (j += SIMD_WIDTH) {
            const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
            const vb: Vec4 = tiny_weight_row_buf[j..][0..SIMD_WIDTH].*;
            sum += @reduce(.Add, va * vb);
        }
        output[i] = sum;
    }
}

fn tinyLinearLayer128to512(
    input: *const [TINY_EMBED_DIM]f32,
    w_idx: usize,
    b_idx: usize,
    output: *[TINY_MLP_DIM]f32,
) void {
    for (0..TINY_MLP_DIM) |i| {
        tinyGetWeightRowF32(w_idx, i, TINY_EMBED_DIM, tiny_weight_row_buf[0..TINY_EMBED_DIM]);
        var sum: f32 = tinyReadWeight(b_idx, i);
        var j: usize = 0;
        while (j < TINY_EMBED_DIM) : (j += SIMD_WIDTH) {
            const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
            const vb: Vec4 = tiny_weight_row_buf[j..][0..SIMD_WIDTH].*;
            sum += @reduce(.Add, va * vb);
        }
        output[i] = sum;
    }
}

fn tinyLinearLayer512to128(
    input: *const [TINY_MLP_DIM]f32,
    w_idx: usize,
    b_idx: usize,
    output: *[TINY_EMBED_DIM]f32,
) void {
    for (0..TINY_EMBED_DIM) |i| {
        tinyGetWeightRowF32(w_idx, i, TINY_MLP_DIM, &tiny_weight_row_buf);
        var sum: f32 = tinyReadWeight(b_idx, i);
        var j: usize = 0;
        while (j < TINY_MLP_DIM) : (j += SIMD_WIDTH) {
            const va: Vec4 = input[j..][0..SIMD_WIDTH].*;
            const vb: Vec4 = tiny_weight_row_buf[j..][0..SIMD_WIDTH].*;
            sum += @reduce(.Add, va * vb);
        }
        output[i] = sum;
    }
}

fn tinyLayerNorm(input: []f32, seq_len: usize, weight_idx: usize, bias_idx: usize) void {
    const eps: f32 = 1e-12;

    for (0..seq_len) |pos| {
        var h: *[TINY_EMBED_DIM]f32 = @ptrCast(input[pos * TINY_EMBED_DIM ..][0..TINY_EMBED_DIM]);

        var sum_vec: Vec4 = @splat(0);
        var i: usize = 0;
        while (i < TINY_EMBED_DIM) : (i += SIMD_WIDTH) {
            sum_vec += h[i..][0..SIMD_WIDTH].*;
        }
        const mean = @reduce(.Add, sum_vec) / @as(f32, TINY_EMBED_DIM);

        const mean_vec: Vec4 = @splat(mean);
        var var_vec: Vec4 = @splat(0);
        i = 0;
        while (i < TINY_EMBED_DIM) : (i += SIMD_WIDTH) {
            const diff = h[i..][0..SIMD_WIDTH].* - mean_vec;
            var_vec += diff * diff;
        }
        const variance = @reduce(.Add, var_vec) / @as(f32, TINY_EMBED_DIM);
        const inv_std = 1.0 / @sqrt(variance + eps);

        tinyGetWeightRowF32(weight_idx, 0, TINY_EMBED_DIM, tiny_weight_row_buf[0..TINY_EMBED_DIM]);
        var bias_buf: *[TINY_EMBED_DIM]f32 = @ptrCast(tiny_scratch_ln[0..TINY_EMBED_DIM]);
        tinyGetWeightRowF32(bias_idx, 0, TINY_EMBED_DIM, bias_buf);

        const inv_std_vec: Vec4 = @splat(inv_std);
        i = 0;
        while (i < TINY_EMBED_DIM) : (i += SIMD_WIDTH) {
            const x = h[i..][0..SIMD_WIDTH].*;
            const w: Vec4 = tiny_weight_row_buf[i..][0..SIMD_WIDTH].*;
            const b: Vec4 = bias_buf[i..][0..SIMD_WIDTH].*;
            h[i..][0..SIMD_WIDTH].* = (x - mean_vec) * inv_std_vec * w + b;
        }
    }
}

fn tinyMultiHeadAttention(input: []f32, seq_len: usize, layer: usize) void {
    const q_w_idx = tiny_layer_q_weight_idx[layer] orelse return;
    const q_b_idx = tiny_layer_q_bias_idx[layer] orelse return;
    const k_w_idx = tiny_layer_k_weight_idx[layer] orelse return;
    const k_b_idx = tiny_layer_k_bias_idx[layer] orelse return;
    const v_w_idx = tiny_layer_v_weight_idx[layer] orelse return;
    const v_b_idx = tiny_layer_v_bias_idx[layer] orelse return;
    const out_w_idx = tiny_layer_out_weight_idx[layer] orelse return;
    const out_b_idx = tiny_layer_out_bias_idx[layer] orelse return;
    const ln_w_idx = tiny_layer_attn_ln_weight_idx[layer] orelse return;
    const ln_b_idx = tiny_layer_attn_ln_bias_idx[layer] orelse return;

    const scale: f32 = 1.0 / @sqrt(@as(f32, TINY_HEAD_DIM));

    for (0..seq_len) |pos| {
        const h: *const [TINY_EMBED_DIM]f32 = @ptrCast(input[pos * TINY_EMBED_DIM ..][0..TINY_EMBED_DIM]);
        const q: *[TINY_EMBED_DIM]f32 = @ptrCast(tiny_scratch_q[pos * TINY_EMBED_DIM ..][0..TINY_EMBED_DIM]);
        const k: *[TINY_EMBED_DIM]f32 = @ptrCast(tiny_scratch_k[pos * TINY_EMBED_DIM ..][0..TINY_EMBED_DIM]);
        const v: *[TINY_EMBED_DIM]f32 = @ptrCast(tiny_scratch_v[pos * TINY_EMBED_DIM ..][0..TINY_EMBED_DIM]);

        tinyLinearLayer128to128(h, q_w_idx, q_b_idx, q);
        tinyLinearLayer128to128(h, k_w_idx, k_b_idx, k);
        tinyLinearLayer128to128(h, v_w_idx, v_b_idx, v);
    }

    for (0..TINY_NUM_HEADS) |head| {
        const head_offset = head * TINY_HEAD_DIM;

        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                var sum: f32 = 0;
                var d: usize = 0;
                while (d < TINY_HEAD_DIM) : (d += SIMD_WIDTH) {
                    const qi: Vec4 = tiny_scratch_q[i * TINY_EMBED_DIM + head_offset + d ..][0..SIMD_WIDTH].*;
                    const kj: Vec4 = tiny_scratch_k[j * TINY_EMBED_DIM + head_offset + d ..][0..SIMD_WIDTH].*;
                    sum += @reduce(.Add, qi * kj);
                }
                tiny_scratch_attn[head * seq_len * seq_len + i * seq_len + j] = sum * scale;
            }
        }

        for (0..seq_len) |i| {
            const row_start = head * seq_len * seq_len + i * seq_len;

            var max_val: f32 = tiny_scratch_attn[row_start];
            for (1..seq_len) |j| {
                if (tiny_scratch_attn[row_start + j] > max_val) max_val = tiny_scratch_attn[row_start + j];
            }

            var sum: f32 = 0;
            for (0..seq_len) |j| {
                tiny_scratch_attn[row_start + j] = @exp(tiny_scratch_attn[row_start + j] - max_val);
                sum += tiny_scratch_attn[row_start + j];
            }

            const inv_sum = 1.0 / sum;
            for (0..seq_len) |j| {
                tiny_scratch_attn[row_start + j] *= inv_sum;
            }
        }
    }

    for (0..seq_len) |pos| {
        for (0..TINY_EMBED_DIM) |d| {
            tiny_scratch_attn_output[pos * TINY_EMBED_DIM + d] = 0;
        }

        for (0..TINY_NUM_HEADS) |head| {
            const head_offset = head * TINY_HEAD_DIM;

            for (0..seq_len) |j| {
                const attn_weight = tiny_scratch_attn[head * seq_len * seq_len + pos * seq_len + j];
                for (0..TINY_HEAD_DIM) |d| {
                    tiny_scratch_attn_output[pos * TINY_EMBED_DIM + head_offset + d] +=
                        attn_weight * tiny_scratch_v[j * TINY_EMBED_DIM + head_offset + d];
                }
            }
        }

        const attn_vec: *const [TINY_EMBED_DIM]f32 = @ptrCast(tiny_scratch_attn_output[pos * TINY_EMBED_DIM ..][0..TINY_EMBED_DIM]);
        tinyLinearLayer128to128(attn_vec, out_w_idx, out_b_idx, &tiny_scratch_proj_out);

        for (0..TINY_EMBED_DIM) |d| {
            input[pos * TINY_EMBED_DIM + d] += tiny_scratch_proj_out[d];
        }
    }

    tinyLayerNorm(input, seq_len, ln_w_idx, ln_b_idx);
}

fn tinyFFNBlock(input: []f32, seq_len: usize, layer: usize) void {
    const up_w_idx = tiny_layer_ffn_up_weight_idx[layer] orelse return;
    const up_b_idx = tiny_layer_ffn_up_bias_idx[layer] orelse return;
    const down_w_idx = tiny_layer_ffn_down_weight_idx[layer] orelse return;
    const down_b_idx = tiny_layer_ffn_down_bias_idx[layer] orelse return;
    const ln_w_idx = tiny_layer_ffn_ln_weight_idx[layer] orelse return;
    const ln_b_idx = tiny_layer_ffn_ln_bias_idx[layer] orelse return;

    for (0..seq_len) |pos| {
        const h: *const [TINY_EMBED_DIM]f32 = @ptrCast(input[pos * TINY_EMBED_DIM ..][0..TINY_EMBED_DIM]);

        tinyLinearLayer128to512(h, up_w_idx, up_b_idx, &tiny_scratch_mlp_hidden);

        const sqrt2_inv: f32 = 1.0 / @sqrt(2.0);
        var i: usize = 0;
        while (i < TINY_MLP_DIM) : (i += SIMD_WIDTH) {
            const x: Vec4 = tiny_scratch_mlp_hidden[i..][0..SIMD_WIDTH].*;
            var erf_val: Vec4 = undefined;
            inline for (0..SIMD_WIDTH) |k| {
                erf_val[k] = erf(x[k] * sqrt2_inv);
            }
            const half: Vec4 = @splat(0.5);
            const one: Vec4 = @splat(1.0);
            tiny_scratch_mlp_hidden[i..][0..SIMD_WIDTH].* = x * half * (one + erf_val);
        }

        tinyLinearLayer512to128(&tiny_scratch_mlp_hidden, down_w_idx, down_b_idx, &tiny_scratch_proj_out);

        for (0..TINY_EMBED_DIM) |d| {
            input[pos * TINY_EMBED_DIM + d] += tiny_scratch_proj_out[d];
        }
    }

    tinyLayerNorm(input, seq_len, ln_w_idx, ln_b_idx);
}

// ============================================================================
// WASM Exports
// ============================================================================

pub export fn tinybert_init() i32 {
    tiny_initialized = true;
    tiny_model_loaded = false;
    tiny_classification_output = 0;
    return 0;
}

pub export fn tinybert_get_query_buffer() [*]u8 {
    return &tiny_query_buffer;
}

pub export fn tinybert_get_query_buffer_size() usize {
    return tiny_query_buffer.len;
}

pub export fn tinybert_get_doc_buffer() [*]u8 {
    return &tiny_doc_buffer;
}

pub export fn tinybert_get_doc_buffer_size() usize {
    return tiny_doc_buffer.len;
}

pub export fn tinybert_get_score() f32 {
    return tiny_classification_output;
}

pub export fn tinybert_alloc_model_buffer(size: usize) usize {
    const page_size: usize = 65536;
    const pages_needed = (size + page_size - 1) / page_size;

    const current_pages = @wasmMemorySize(0);
    const current_size = current_pages * page_size;

    const result = @wasmMemoryGrow(0, pages_needed);
    if (result == @as(usize, @bitCast(@as(isize, -1)))) {
        return 0;
    }

    const ptr: [*]u8 = @ptrFromInt(current_size);
    tiny_model_buffer = ptr;
    tiny_model_size = size;
    return current_size;
}

pub export fn tinybert_weights_loaded() i32 {
    return if (tiny_model_loaded) 1 else 0;
}

pub export fn tinybert_load_model(size: usize) i32 {
    const model_data = tiny_model_buffer orelse return -1;
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

            if (atype == GGUF_TYPE_STRING and alen <= TINY_VOCAB_SIZE) {
                tiny_vocab_data = data.ptr + pos;
                tiny_vocab_count = alen;

                var str_pos: u32 = 0;
                for (0..alen) |i| {
                    tiny_vocab_string_offsets[i] = str_pos;
                    const slen: u32 = @intCast(readU64LE(data, pos));
                    pos += 8;
                    str_pos += slen + 8;
                    pos += slen;
                }
                tiny_vocab_string_offsets[alen] = str_pos;
            } else {
                ggufSkipValue(data, &pos, vtype);
            }
        } else {
            ggufSkipValue(data, &pos, vtype);
        }
    }

    tiny_n_tensors_loaded = @min(n_tensors, MAX_TENSORS);
    for (0..tiny_n_tensors_loaded) |i| {
        const name = ggufReadString(data, &pos);
        const name_len = @min(name.len, 95);
        @memcpy(tiny_tensor_names[i][0..name_len], name[0..name_len]);
        tiny_tensor_name_lens[i] = name_len;

        if (pos + 4 > data.len) return -7;
        const n_dims = readU32LE(data, pos);
        pos += 4;
        tiny_tensor_n_dims[i] = n_dims;

        for (0..@min(n_dims, 4)) |d| {
            if (pos + 8 > data.len) return -8;
            tiny_tensor_dims[i][d] = readU64LE(data, pos);
            pos += 8;
        }

        if (pos + 12 > data.len) return -9;
        tiny_tensor_types[i] = readU32LE(data, pos);
        pos += 4;
        tiny_tensor_offsets[i] = readU64LE(data, pos);
        pos += 8;
    }

    tiny_gguf_data_offset = (pos + 31) & ~@as(usize, 31);

    // Load embedding weights
    tiny_word_emb_idx = tinyFindTensor("bert.embeddings.word_embeddings.weight");
    tiny_pos_emb_idx = tinyFindTensor("bert.embeddings.position_embeddings.weight");
    tiny_token_type_emb_idx = tinyFindTensor("bert.embeddings.token_type_embeddings.weight");
    tiny_emb_ln_weight_idx = tinyFindTensor("bert.embeddings.LayerNorm.weight");
    tiny_emb_ln_bias_idx = tinyFindTensor("bert.embeddings.LayerNorm.bias");

    // Load layer weights (2 layers)
    for (0..TINY_NUM_LAYERS) |layer| {
        var name_buf: [128]u8 = undefined;

        const q_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.query.weight", .{layer}) catch continue;
        tiny_layer_q_weight_idx[layer] = tinyFindTensor(q_w);

        const q_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.query.bias", .{layer}) catch continue;
        tiny_layer_q_bias_idx[layer] = tinyFindTensor(q_b);

        const k_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.key.weight", .{layer}) catch continue;
        tiny_layer_k_weight_idx[layer] = tinyFindTensor(k_w);

        const k_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.key.bias", .{layer}) catch continue;
        tiny_layer_k_bias_idx[layer] = tinyFindTensor(k_b);

        const v_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.value.weight", .{layer}) catch continue;
        tiny_layer_v_weight_idx[layer] = tinyFindTensor(v_w);

        const v_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.self.value.bias", .{layer}) catch continue;
        tiny_layer_v_bias_idx[layer] = tinyFindTensor(v_b);

        const out_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.dense.weight", .{layer}) catch continue;
        tiny_layer_out_weight_idx[layer] = tinyFindTensor(out_w);

        const out_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.dense.bias", .{layer}) catch continue;
        tiny_layer_out_bias_idx[layer] = tinyFindTensor(out_b);

        const attn_ln_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer}) catch continue;
        tiny_layer_attn_ln_weight_idx[layer] = tinyFindTensor(attn_ln_w);

        const attn_ln_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer}) catch continue;
        tiny_layer_attn_ln_bias_idx[layer] = tinyFindTensor(attn_ln_b);

        const up_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.intermediate.dense.weight", .{layer}) catch continue;
        tiny_layer_ffn_up_weight_idx[layer] = tinyFindTensor(up_w);

        const up_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.intermediate.dense.bias", .{layer}) catch continue;
        tiny_layer_ffn_up_bias_idx[layer] = tinyFindTensor(up_b);

        const down_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.dense.weight", .{layer}) catch continue;
        tiny_layer_ffn_down_weight_idx[layer] = tinyFindTensor(down_w);

        const down_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.dense.bias", .{layer}) catch continue;
        tiny_layer_ffn_down_bias_idx[layer] = tinyFindTensor(down_b);

        const ffn_ln_w = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.LayerNorm.weight", .{layer}) catch continue;
        tiny_layer_ffn_ln_weight_idx[layer] = tinyFindTensor(ffn_ln_w);

        const ffn_ln_b = std.fmt.bufPrint(&name_buf, "bert.encoder.layer.{d}.output.LayerNorm.bias", .{layer}) catch continue;
        tiny_layer_ffn_ln_bias_idx[layer] = tinyFindTensor(ffn_ln_b);
    }

    // Load classification head (required for cross-encoder)
    tiny_cls_weight_idx = tinyFindTensor("classifier.weight");
    if (tiny_cls_weight_idx == null) {
        tiny_cls_weight_idx = tinyFindTensor("classifier.dense.weight");
    }
    if (tiny_cls_weight_idx == null) {
        tiny_cls_weight_idx = tinyFindTensor("classifier.linear.weight");
    }

    tiny_cls_bias_idx = tinyFindTensor("classifier.bias");
    if (tiny_cls_bias_idx == null) {
        tiny_cls_bias_idx = tinyFindTensor("classifier.dense.bias");
    }
    if (tiny_cls_bias_idx == null) {
        tiny_cls_bias_idx = tinyFindTensor("classifier.linear.bias");
    }

    // Verify required weights
    if (tiny_word_emb_idx == null) return -10;
    if (tiny_pos_emb_idx == null) return -11;
    if (tiny_emb_ln_weight_idx == null) return -12;
    if (tiny_cls_weight_idx == null) return -13; // Classification head required
    if (tiny_cls_bias_idx == null) return -14;

    tiny_model_loaded = true;
    return 0;
}

/// Score a (query, document) pair using TinyBERT cross-encoder.
/// Returns 0 on success, negative error code on failure.
pub export fn tinybert_cross_encode(query_len: usize, doc_len: usize) i32 {
    if (!tiny_initialized) return -1;
    if (!tiny_model_loaded) return -2;
    if (query_len == 0 or query_len > tiny_query_buffer.len) return -3;
    if (doc_len == 0 or doc_len > tiny_doc_buffer.len) return -4;

    if (tiny_word_emb_idx == null) return -5;
    if (tiny_pos_emb_idx == null) return -6;
    if (tiny_token_type_emb_idx == null) return -7;
    if (tiny_emb_ln_weight_idx == null) return -8;
    if (tiny_emb_ln_bias_idx == null) return -9;

    const cls_w_idx = tiny_cls_weight_idx orelse return -10;
    const cls_b_idx = tiny_cls_bias_idx orelse return -11;

    // Tokenize query-document pair with segment IDs
    const seq_len = tinyTokenizePair(
        tiny_query_buffer[0..query_len],
        tiny_doc_buffer[0..doc_len],
        &tiny_tokens,
        &tiny_attention_mask,
        &tiny_token_types,
    );

    if (seq_len == 0) return -12;

    // Build embeddings with token types
    tinyBuildEmbeddings(seq_len);

    // Run BERT encoder (2 layers)
    tinyRunEncoder(seq_len);

    // Extract CLS token embedding (position 0)
    const cls_embedding: *const [TINY_EMBED_DIM]f32 = @ptrCast(tiny_scratch_hidden[0..TINY_EMBED_DIM]);

    // Apply classification head: score = sigmoid(W @ cls + b)
    tinyGetWeightRowF32(cls_w_idx, 0, TINY_EMBED_DIM, tiny_weight_row_buf[0..TINY_EMBED_DIM]);

    var logit: f32 = tinyReadWeight(cls_b_idx, 0);

    var j: usize = 0;
    while (j < TINY_EMBED_DIM) : (j += SIMD_WIDTH) {
        const va: Vec4 = cls_embedding[j..][0..SIMD_WIDTH].*;
        const vb: Vec4 = tiny_weight_row_buf[j..][0..SIMD_WIDTH].*;
        logit += @reduce(.Add, va * vb);
    }

    // Apply sigmoid
    tiny_classification_output = 1.0 / (1.0 + @exp(-logit));

    return 0;
}

// ============================================================================
// Tests
// ============================================================================

test "tinybert_model: initialization" {
    tiny_initialized = false;
    tiny_model_loaded = false;
    const result = tinybert_init();
    try std.testing.expectEqual(@as(i32, 0), result);
    try std.testing.expect(tiny_initialized);
    try std.testing.expect(!tiny_model_loaded);
}

test "tinybert_model: buffer access" {
    const query_buf = tinybert_get_query_buffer();
    try std.testing.expect(query_buf != undefined);

    const query_size = tinybert_get_query_buffer_size();
    try std.testing.expectEqual(@as(usize, 512), query_size);

    const doc_buf = tinybert_get_doc_buffer();
    try std.testing.expect(doc_buf != undefined);

    const doc_size = tinybert_get_doc_buffer_size();
    try std.testing.expectEqual(@as(usize, 512), doc_size);
}

test "tinybert_model: weights not loaded" {
    tiny_initialized = true;
    tiny_model_loaded = false;
    const result = tinybert_weights_loaded();
    try std.testing.expectEqual(@as(i32, 0), result);
}
