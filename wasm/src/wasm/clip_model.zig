//! CLIP Text Encoder for WASM
//!
//! OpenAI CLIP ViT-B/32 text encoder implementation.
//! Encodes text to 512-dimensional L2-normalized embeddings.

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
// CLIP Model Constants (ViT-B/32)
// ============================================================================

const CLIP_VOCAB_SIZE: usize = 49408;
const CLIP_MAX_SEQ_LEN: usize = 77;
const CLIP_EMBED_DIM: usize = 512;
const CLIP_NUM_HEADS: usize = 8;
const CLIP_NUM_LAYERS: usize = 12;
const CLIP_MLP_DIM: usize = 2048;
const CLIP_HEAD_DIM: usize = CLIP_EMBED_DIM / CLIP_NUM_HEADS; // 64

// Comptime assertions for SIMD alignment
comptime {
    if (CLIP_EMBED_DIM % SIMD_WIDTH != 0) @compileError("CLIP_EMBED_DIM must be divisible by SIMD_WIDTH");
    if (CLIP_HEAD_DIM % SIMD_WIDTH != 0) @compileError("CLIP_HEAD_DIM must be divisible by SIMD_WIDTH");
    if (CLIP_MLP_DIM % SIMD_WIDTH != 0) @compileError("CLIP_MLP_DIM must be divisible by SIMD_WIDTH");
}

// ============================================================================
// CLIP State
// ============================================================================

var clip_initialized: bool = false;
var clip_model_loaded: bool = false;

// Buffers for CLIP
var clip_text_buffer: [1024]u8 = undefined;
var clip_output_buffer: [CLIP_EMBED_DIM]f32 = undefined;

// Model weights storage (allocated from model buffer)
var clip_model_buffer: ?[*]u8 = null;
var clip_model_size: usize = 0;

// Weight tensor indices (set after model load)
var token_embedding_idx: ?usize = null;
var position_embedding_idx: ?usize = null;
var ln_final_weight_idx: ?usize = null;
var ln_final_bias_idx: ?usize = null;
var text_projection_idx: ?usize = null;

// Per-layer weight indices (12 layers)
var layer_ln1_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_ln1_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_q_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_q_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_k_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_k_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_v_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_v_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_out_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_attn_out_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_ln2_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_ln2_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc1_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc1_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc2_weight_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;
var layer_mlp_fc2_bias_idx: [CLIP_NUM_LAYERS]?usize = [_]?usize{null} ** CLIP_NUM_LAYERS;

// Tokenizer vocab
var vocab_data: ?[*]const u8 = null;
var vocab_string_offsets: [CLIP_VOCAB_SIZE + 1]u32 = undefined;
var vocab_count: usize = 0;

// Scratch buffers for inference (statically allocated)
var scratch_hidden: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_q: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_k: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_v: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;
var scratch_attn: [CLIP_NUM_HEADS * CLIP_MAX_SEQ_LEN * CLIP_MAX_SEQ_LEN]f32 = undefined;
var scratch_mlp: [CLIP_MAX_SEQ_LEN * CLIP_MLP_DIM]f32 = undefined;
var scratch_ln: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;

// Tensor info storage
const MAX_TENSORS: usize = 256;
var tensor_names: [MAX_TENSORS][64]u8 = undefined;
var tensor_name_lens: [MAX_TENSORS]usize = undefined;
var tensor_offsets: [MAX_TENSORS]u64 = undefined;
var tensor_types: [MAX_TENSORS]u32 = undefined;
var tensor_dims: [MAX_TENSORS][4]u64 = undefined;
var tensor_n_dims: [MAX_TENSORS]u32 = undefined;
var n_tensors_loaded: usize = 0;
var gguf_data_offset: usize = 0;

// Scratch buffer for weight row loading
var weight_row_buf: [CLIP_MLP_DIM]f32 = undefined;

// Debug state
var debug_tokens: [CLIP_MAX_SEQ_LEN]u32 = undefined;
var debug_token_count: usize = 0;
var debug_hidden: [CLIP_EMBED_DIM]f32 = undefined;
var debug_stage: usize = 0;
var debug_after_embedding: [CLIP_EMBED_DIM]f32 = undefined;
var debug_after_ln1: [CLIP_EMBED_DIM]f32 = undefined;
var debug_after_attn: [CLIP_EMBED_DIM]f32 = undefined;
var debug_after_ln2: [CLIP_EMBED_DIM]f32 = undefined;
var debug_after_fc1: [8]f32 = undefined;
var debug_after_fc2: [8]f32 = undefined;
var debug_after_layer0: [CLIP_EMBED_DIM]f32 = undefined;
var debug_ln2_w: [8]f32 = undefined;
var debug_ln2_b: [8]f32 = undefined;
var debug_pre_ln2: [8]f32 = undefined;
var debug_ln2_mean: f32 = 0;
var debug_ln2_std: f32 = 0;
var debug_pre_norm: [8]f32 = undefined;
var debug_pre_norm_norm: f32 = 0;

// ============================================================================
// SIMD Helpers
// ============================================================================

inline fn simdDot(comptime N: usize, a: *const [N]f32, b: *const [N]f32) f32 {
    comptime {
        if (N % SIMD_WIDTH != 0) @compileError("N must be divisible by SIMD_WIDTH");
    }
    var acc: Vec4 = @splat(0);
    inline for (0..N / SIMD_WIDTH) |i| {
        const va: Vec4 = a[i * SIMD_WIDTH ..][0..SIMD_WIDTH].*;
        const vb: Vec4 = b[i * SIMD_WIDTH ..][0..SIMD_WIDTH].*;
        acc += va * vb;
    }
    return @reduce(.Add, acc);
}

inline fn simdAdd(comptime N: usize, dst: *[N]f32, src: *const [N]f32) void {
    comptime {
        if (N % SIMD_WIDTH != 0) @compileError("N must be divisible by SIMD_WIDTH");
    }
    inline for (0..N / SIMD_WIDTH) |i| {
        const vd: Vec4 = dst[i * SIMD_WIDTH ..][0..SIMD_WIDTH].*;
        const vs: Vec4 = src[i * SIMD_WIDTH ..][0..SIMD_WIDTH].*;
        dst[i * SIMD_WIDTH ..][0..SIMD_WIDTH].* = vd + vs;
    }
}

inline fn simdScalarMulAdd(comptime N: usize, dst: *[N]f32, scalar: f32, src: *const [N]f32) void {
    comptime {
        if (N % SIMD_WIDTH != 0) @compileError("N must be divisible by SIMD_WIDTH");
    }
    const vs: Vec4 = @splat(scalar);
    inline for (0..N / SIMD_WIDTH) |i| {
        const vd: Vec4 = dst[i * SIMD_WIDTH ..][0..SIMD_WIDTH].*;
        const vsrc: Vec4 = src[i * SIMD_WIDTH ..][0..SIMD_WIDTH].*;
        dst[i * SIMD_WIDTH ..][0..SIMD_WIDTH].* = vd + vs * vsrc;
    }
}

inline fn simdZero(comptime N: usize, dst: *[N]f32) void {
    comptime {
        if (N % SIMD_WIDTH != 0) @compileError("N must be divisible by SIMD_WIDTH");
    }
    const zero: Vec4 = @splat(0);
    inline for (0..N / SIMD_WIDTH) |i| {
        dst[i * SIMD_WIDTH ..][0..SIMD_WIDTH].* = zero;
    }
}

inline fn simdCopy(comptime N: usize, dst: *[N]f32, src: *const [N]f32) void {
    comptime {
        if (N % SIMD_WIDTH != 0) @compileError("N must be divisible by SIMD_WIDTH");
    }
    inline for (0..N / SIMD_WIDTH) |i| {
        dst[i * SIMD_WIDTH ..][0..SIMD_WIDTH].* = src[i * SIMD_WIDTH ..][0..SIMD_WIDTH].*;
    }
}

// ============================================================================
// Tensor Access
// ============================================================================

fn findTensor(name: []const u8) ?usize {
    for (0..n_tensors_loaded) |i| {
        if (strEql(tensor_names[i][0..tensor_name_lens[i]], name)) {
            return i;
        }
    }
    return null;
}

fn readWeight(idx: usize, i: usize) f32 {
    const model_data = clip_model_buffer orelse return 0;
    const offset: usize = gguf_data_offset + @as(usize, @intCast(tensor_offsets[idx]));

    if (tensor_types[idx] == GGML_TYPE_F16) {
        const ptr: [*]const u16 = @ptrCast(@alignCast(model_data + offset));
        return f16ToF32(ptr[i]);
    } else {
        const ptr: [*]const f32 = @ptrCast(@alignCast(model_data + offset));
        return ptr[i];
    }
}

fn getWeightRowF32(idx: usize, row: usize, row_len: usize, buf: []f32) void {
    const model_data = clip_model_buffer orelse return;
    const offset: usize = gguf_data_offset + @as(usize, @intCast(tensor_offsets[idx]));
    const row_offset = row * row_len;

    if (tensor_types[idx] == GGML_TYPE_F16) {
        const ptr: [*]const u16 = @ptrCast(@alignCast(model_data + offset));
        var i: usize = 0;
        while (i + 4 <= row_len) : (i += 4) {
            buf[i] = f16ToF32(ptr[row_offset + i]);
            buf[i + 1] = f16ToF32(ptr[row_offset + i + 1]);
            buf[i + 2] = f16ToF32(ptr[row_offset + i + 2]);
            buf[i + 3] = f16ToF32(ptr[row_offset + i + 3]);
        }
        while (i < row_len) : (i += 1) {
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

fn getVocabToken(idx: usize) []const u8 {
    if (vocab_data == null or idx >= vocab_count) return "";
    const start = vocab_string_offsets[idx];
    const vdata = vocab_data.?;
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
    return vdata[start + 8 .. start + 8 + len];
}

fn startsWithToken(text: []const u8, token: []const u8) bool {
    if (token.len == 0 or token.len > text.len) return false;
    for (0..token.len) |i| {
        var tc = token[i];
        var xc = text[i];
        if (tc >= 'A' and tc <= 'Z') tc = tc + 32;
        if (xc >= 'A' and xc <= 'Z') xc = xc + 32;
        if (tc != xc) return false;
    }
    return true;
}

fn tokenize(text: []const u8, tokens: []u32) usize {
    var n_tokens: usize = 0;
    var text_pos: usize = 0;

    // Start token
    tokens[n_tokens] = 49406;
    n_tokens += 1;

    while (text_pos < text.len and n_tokens < CLIP_MAX_SEQ_LEN - 1) {
        if (text[text_pos] == 0) break;

        if (text[text_pos] == ' ') {
            text_pos += 1;
            continue;
        }

        var best_len: usize = 0;
        var best_id: u32 = 0;

        for (0..vocab_count) |i| {
            const tok = getVocabToken(i);
            if (tok.len > 0 and tok.len > best_len) {
                var tok_text = tok;
                var is_word_end = false;
                if (tok.len >= 4 and tok[tok.len - 4] == '<' and tok[tok.len - 3] == '/' and tok[tok.len - 2] == 'w' and tok[tok.len - 1] == '>') {
                    tok_text = tok[0 .. tok.len - 4];
                    is_word_end = true;
                }

                if (startsWithToken(text[text_pos..], tok_text)) {
                    if (is_word_end) {
                        const next_pos = text_pos + tok_text.len;
                        if (next_pos >= text.len or text[next_pos] == ' ' or text[next_pos] == 0) {
                            best_len = tok_text.len;
                            best_id = @intCast(i);
                        }
                    } else {
                        best_len = tok_text.len;
                        best_id = @intCast(i);
                    }
                }
            }
        }

        if (best_len > 0) {
            tokens[n_tokens] = best_id;
            n_tokens += 1;
            text_pos += best_len;
        } else {
            text_pos += 1;
        }
    }

    // End token
    tokens[n_tokens] = 49407;
    n_tokens += 1;

    const final_len = n_tokens;
    while (n_tokens < CLIP_MAX_SEQ_LEN) {
        tokens[n_tokens] = 49407;
        n_tokens += 1;
    }

    return final_len;
}

// ============================================================================
// Neural Network Layers
// ============================================================================

fn linearLayerSimd(
    comptime in_dim: usize,
    comptime out_dim: usize,
    input: *const [in_dim]f32,
    w_idx: usize,
    b_idx: usize,
    output: *[out_dim]f32,
    weight_buf_local: *[in_dim]f32,
) void {
    for (0..out_dim) |i| {
        getWeightRowF32(w_idx, i, in_dim, weight_buf_local);
        output[i] = readWeight(b_idx, i) + simdDot(in_dim, input, weight_buf_local);
    }
}

fn layerNormInPlace(hidden: []f32, seq_len: usize, weight_idx: usize, bias_idx: usize) void {
    const eps: f32 = 1e-5;

    for (0..seq_len) |pos| {
        var h: *[CLIP_EMBED_DIM]f32 = @ptrCast(hidden[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);

        var sum_vec: Vec4 = @splat(0);
        var i: usize = 0;
        while (i < CLIP_EMBED_DIM) : (i += SIMD_WIDTH) {
            sum_vec += h[i..][0..SIMD_WIDTH].*;
        }
        const mean = @reduce(.Add, sum_vec) / @as(f32, CLIP_EMBED_DIM);

        const mean_vec: Vec4 = @splat(mean);
        var var_vec: Vec4 = @splat(0);
        i = 0;
        while (i < CLIP_EMBED_DIM) : (i += SIMD_WIDTH) {
            const diff = h[i..][0..SIMD_WIDTH].* - mean_vec;
            var_vec += diff * diff;
        }
        const variance = @reduce(.Add, var_vec) / @as(f32, CLIP_EMBED_DIM);
        const inv_std = 1.0 / @sqrt(variance + eps);

        var weight_buf_ln: [CLIP_EMBED_DIM]f32 = undefined;
        var bias_buf: [CLIP_EMBED_DIM]f32 = undefined;
        getWeightRowF32(weight_idx, 0, CLIP_EMBED_DIM, &weight_buf_ln);
        getWeightRowF32(bias_idx, 0, CLIP_EMBED_DIM, &bias_buf);

        const inv_std_vec: Vec4 = @splat(inv_std);
        i = 0;
        while (i < CLIP_EMBED_DIM) : (i += SIMD_WIDTH) {
            const x = h[i..][0..SIMD_WIDTH].*;
            const w: Vec4 = weight_buf_ln[i..][0..SIMD_WIDTH].*;
            const b: Vec4 = bias_buf[i..][0..SIMD_WIDTH].*;
            h[i..][0..SIMD_WIDTH].* = (x - mean_vec) * inv_std_vec * w + b;
        }
    }
}

fn multiHeadAttention(
    input: []f32,
    residual: []f32,
    seq_len: usize,
    layer: usize,
    q_out: []f32,
    k_out: []f32,
    v_out: []f32,
    attn_out: []f32,
) void {
    const q_w_idx = layer_attn_q_weight_idx[layer] orelse return;
    const q_b_idx = layer_attn_q_bias_idx[layer] orelse return;
    const k_w_idx = layer_attn_k_weight_idx[layer] orelse return;
    const k_b_idx = layer_attn_k_bias_idx[layer] orelse return;
    const v_w_idx = layer_attn_v_weight_idx[layer] orelse return;
    const v_b_idx = layer_attn_v_bias_idx[layer] orelse return;
    const out_w_idx = layer_attn_out_weight_idx[layer] orelse return;
    const out_b_idx = layer_attn_out_bias_idx[layer] orelse return;

    const scale: f32 = 1.0 / @sqrt(@as(f32, CLIP_HEAD_DIM));

    for (0..seq_len) |pos| {
        const h: *const [CLIP_EMBED_DIM]f32 = @ptrCast(input[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);
        const q: *[CLIP_EMBED_DIM]f32 = @ptrCast(q_out[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);
        const k: *[CLIP_EMBED_DIM]f32 = @ptrCast(k_out[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);
        const v: *[CLIP_EMBED_DIM]f32 = @ptrCast(v_out[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);

        linearLayerSimd(CLIP_EMBED_DIM, CLIP_EMBED_DIM, h, q_w_idx, q_b_idx, q, weight_row_buf[0..CLIP_EMBED_DIM]);
        linearLayerSimd(CLIP_EMBED_DIM, CLIP_EMBED_DIM, h, k_w_idx, k_b_idx, k, weight_row_buf[0..CLIP_EMBED_DIM]);
        linearLayerSimd(CLIP_EMBED_DIM, CLIP_EMBED_DIM, h, v_w_idx, v_b_idx, v, weight_row_buf[0..CLIP_EMBED_DIM]);
    }

    for (0..CLIP_NUM_HEADS) |head| {
        const head_offset = head * CLIP_HEAD_DIM;

        for (0..seq_len) |i| {
            const qi: *const [CLIP_HEAD_DIM]f32 = @ptrCast(q_out[i * CLIP_EMBED_DIM + head_offset ..][0..CLIP_HEAD_DIM]);

            for (0..seq_len) |j| {
                if (j > i) {
                    attn_out[head * seq_len * seq_len + i * seq_len + j] = -1e9;
                } else {
                    const kj: *const [CLIP_HEAD_DIM]f32 = @ptrCast(k_out[j * CLIP_EMBED_DIM + head_offset ..][0..CLIP_HEAD_DIM]);
                    attn_out[head * seq_len * seq_len + i * seq_len + j] = simdDot(CLIP_HEAD_DIM, qi, kj) * scale;
                }
            }
        }

        for (0..seq_len) |i| {
            const row_start = head * seq_len * seq_len + i * seq_len;

            var max_val: f32 = attn_out[row_start];
            for (1..seq_len) |j| {
                if (attn_out[row_start + j] > max_val) max_val = attn_out[row_start + j];
            }

            var sum: f32 = 0;
            for (0..seq_len) |j| {
                attn_out[row_start + j] = @exp(attn_out[row_start + j] - max_val);
                sum += attn_out[row_start + j];
            }

            const inv_sum = 1.0 / sum;
            for (0..seq_len) |j| {
                attn_out[row_start + j] *= inv_sum;
            }
        }
    }

    var attn_output: [CLIP_MAX_SEQ_LEN * CLIP_EMBED_DIM]f32 = undefined;

    for (0..seq_len) |pos| {
        const out_vec: *[CLIP_EMBED_DIM]f32 = @ptrCast(attn_output[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);
        simdZero(CLIP_EMBED_DIM, out_vec);

        for (0..CLIP_NUM_HEADS) |head| {
            const head_offset = head * CLIP_HEAD_DIM;
            const head_out: *[CLIP_HEAD_DIM]f32 = @ptrCast(attn_output[pos * CLIP_EMBED_DIM + head_offset ..][0..CLIP_HEAD_DIM]);

            for (0..seq_len) |j| {
                const attn_weight = attn_out[head * seq_len * seq_len + pos * seq_len + j];
                const vj: *const [CLIP_HEAD_DIM]f32 = @ptrCast(v_out[j * CLIP_EMBED_DIM + head_offset ..][0..CLIP_HEAD_DIM]);
                simdScalarMulAdd(CLIP_HEAD_DIM, head_out, attn_weight, vj);
            }
        }

        const attn_vec: *const [CLIP_EMBED_DIM]f32 = @ptrCast(attn_output[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);
        var proj_out: [CLIP_EMBED_DIM]f32 = undefined;
        linearLayerSimd(CLIP_EMBED_DIM, CLIP_EMBED_DIM, attn_vec, out_w_idx, out_b_idx, &proj_out, weight_row_buf[0..CLIP_EMBED_DIM]);

        const res: *[CLIP_EMBED_DIM]f32 = @ptrCast(residual[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);
        simdAdd(CLIP_EMBED_DIM, res, &proj_out);
    }
}

fn mlpBlock(input: []f32, residual: []f32, seq_len: usize, layer: usize) void {
    const fc1_w_idx = layer_mlp_fc1_weight_idx[layer] orelse return;
    const fc1_b_idx = layer_mlp_fc1_bias_idx[layer] orelse return;
    const fc2_w_idx = layer_mlp_fc2_weight_idx[layer] orelse return;
    const fc2_b_idx = layer_mlp_fc2_bias_idx[layer] orelse return;

    for (0..seq_len) |pos| {
        const h: *const [CLIP_EMBED_DIM]f32 = @ptrCast(input[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);

        var mlp_hidden: [CLIP_MLP_DIM]f32 = undefined;
        linearLayerSimd(CLIP_EMBED_DIM, CLIP_MLP_DIM, h, fc1_w_idx, fc1_b_idx, &mlp_hidden, weight_row_buf[0..CLIP_EMBED_DIM]);

        const sqrt2_inv: f32 = 1.0 / @sqrt(2.0);
        var i: usize = 0;
        while (i < CLIP_MLP_DIM) : (i += SIMD_WIDTH) {
            const x: Vec4 = mlp_hidden[i..][0..SIMD_WIDTH].*;
            var erf_val: Vec4 = undefined;
            inline for (0..SIMD_WIDTH) |k| {
                erf_val[k] = erf(x[k] * sqrt2_inv);
            }
            const half: Vec4 = @splat(0.5);
            const one: Vec4 = @splat(1.0);
            mlp_hidden[i..][0..SIMD_WIDTH].* = x * half * (one + erf_val);
        }

        if (layer == 0 and pos == 0) {
            @memcpy(&debug_after_fc1, mlp_hidden[0..8]);
        }

        var fc2_out: [CLIP_EMBED_DIM]f32 = undefined;
        linearLayerSimd(CLIP_MLP_DIM, CLIP_EMBED_DIM, &mlp_hidden, fc2_w_idx, fc2_b_idx, &fc2_out, &weight_row_buf);

        if (layer == 0 and pos == 0) {
            @memcpy(&debug_after_fc2, fc2_out[0..8]);
        }

        const res: *[CLIP_EMBED_DIM]f32 = @ptrCast(residual[pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM]);
        simdAdd(CLIP_EMBED_DIM, res, &fc2_out);
    }
}

// ============================================================================
// WASM Exports
// ============================================================================

pub export fn clip_init() i32 {
    clip_initialized = true;
    clip_model_loaded = false;

    for (&clip_output_buffer) |*v| {
        v.* = 0;
    }

    return 0;
}

pub export fn clip_get_text_buffer() [*]u8 {
    return &clip_text_buffer;
}

pub export fn clip_get_text_buffer_size() usize {
    return clip_text_buffer.len;
}

pub export fn clip_get_output_buffer() [*]f32 {
    return &clip_output_buffer;
}

pub export fn clip_get_output_dim() usize {
    return CLIP_EMBED_DIM;
}

pub export fn clip_alloc_model_buffer(size: usize) usize {
    const page_size: usize = 65536;
    const pages_needed = (size + page_size - 1) / page_size;

    const current_pages = @wasmMemorySize(0);
    const current_size = current_pages * page_size;

    const result = @wasmMemoryGrow(0, pages_needed);
    if (result == @as(usize, @bitCast(@as(isize, -1)))) {
        return 0;
    }

    const ptr: [*]u8 = @ptrFromInt(current_size);
    clip_model_buffer = ptr;
    clip_model_size = size;
    return current_size;
}

pub export fn clip_weights_loaded() i32 {
    return if (clip_model_loaded) 1 else 0;
}

pub export fn clip_load_model(size: usize) i32 {
    const model_data = clip_model_buffer orelse return -1;
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

            if (atype == GGUF_TYPE_STRING and alen <= CLIP_VOCAB_SIZE) {
                vocab_data = data.ptr + pos;
                vocab_count = alen;

                var str_pos: u32 = 0;
                for (0..alen) |i| {
                    vocab_string_offsets[i] = str_pos;
                    const slen: u32 = @intCast(readU64LE(data, pos));
                    pos += 8;
                    str_pos += slen + 8;
                    pos += slen;
                }
                vocab_string_offsets[alen] = str_pos;
            } else {
                ggufSkipValue(data, &pos, vtype);
            }
        } else {
            ggufSkipValue(data, &pos, vtype);
        }
    }

    n_tensors_loaded = @min(n_tensors, MAX_TENSORS);
    for (0..n_tensors_loaded) |i| {
        const name = ggufReadString(data, &pos);
        const name_len = @min(name.len, 63);
        @memcpy(tensor_names[i][0..name_len], name[0..name_len]);
        tensor_name_lens[i] = name_len;

        if (pos + 4 > data.len) return -7;
        const n_dims = readU32LE(data, pos);
        pos += 4;
        tensor_n_dims[i] = n_dims;

        for (0..@min(n_dims, 4)) |d| {
            if (pos + 8 > data.len) return -8;
            tensor_dims[i][d] = readU64LE(data, pos);
            pos += 8;
        }

        if (pos + 12 > data.len) return -9;
        tensor_types[i] = readU32LE(data, pos);
        pos += 4;
        tensor_offsets[i] = readU64LE(data, pos);
        pos += 8;
    }

    gguf_data_offset = (pos + 31) & ~@as(usize, 31);

    token_embedding_idx = findTensor("t.token_embd.weight");
    position_embedding_idx = findTensor("t.position_embd.weight");
    ln_final_weight_idx = findTensor("t.post_ln.weight");
    ln_final_bias_idx = findTensor("t.post_ln.bias");
    text_projection_idx = findTensor("text_projection.weight");

    for (0..CLIP_NUM_LAYERS) |layer| {
        var name_buf: [64]u8 = undefined;

        const ln1_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln1.weight", .{layer}) catch continue;
        layer_ln1_weight_idx[layer] = findTensor(ln1_w);

        const ln1_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln1.bias", .{layer}) catch continue;
        layer_ln1_bias_idx[layer] = findTensor(ln1_b);

        const q_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_q.weight", .{layer}) catch continue;
        layer_attn_q_weight_idx[layer] = findTensor(q_w);

        const q_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_q.bias", .{layer}) catch continue;
        layer_attn_q_bias_idx[layer] = findTensor(q_b);

        const k_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_k.weight", .{layer}) catch continue;
        layer_attn_k_weight_idx[layer] = findTensor(k_w);

        const k_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_k.bias", .{layer}) catch continue;
        layer_attn_k_bias_idx[layer] = findTensor(k_b);

        const v_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_v.weight", .{layer}) catch continue;
        layer_attn_v_weight_idx[layer] = findTensor(v_w);

        const v_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_v.bias", .{layer}) catch continue;
        layer_attn_v_bias_idx[layer] = findTensor(v_b);

        const out_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_out.weight", .{layer}) catch continue;
        layer_attn_out_weight_idx[layer] = findTensor(out_w);

        const out_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.attn_out.bias", .{layer}) catch continue;
        layer_attn_out_bias_idx[layer] = findTensor(out_b);

        const ln2_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln2.weight", .{layer}) catch continue;
        layer_ln2_weight_idx[layer] = findTensor(ln2_w);

        const ln2_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ln2.bias", .{layer}) catch continue;
        layer_ln2_bias_idx[layer] = findTensor(ln2_b);

        const fc1_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_down.weight", .{layer}) catch continue;
        layer_mlp_fc1_weight_idx[layer] = findTensor(fc1_w);

        const fc1_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_down.bias", .{layer}) catch continue;
        layer_mlp_fc1_bias_idx[layer] = findTensor(fc1_b);

        const fc2_w = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_up.weight", .{layer}) catch continue;
        layer_mlp_fc2_weight_idx[layer] = findTensor(fc2_w);

        const fc2_b = std.fmt.bufPrint(&name_buf, "t.blk.{d}.ffn_up.bias", .{layer}) catch continue;
        layer_mlp_fc2_bias_idx[layer] = findTensor(fc2_b);
    }

    if (token_embedding_idx == null) return -10;
    if (position_embedding_idx == null) return -11;
    if (ln_final_weight_idx == null) return -12;
    if (text_projection_idx == null) return -13;

    clip_model_loaded = true;
    return 0;
}

pub export fn clip_encode_text(text_len: usize) i32 {
    if (!clip_initialized) return -1;
    if (!clip_model_loaded) return -2;
    if (text_len == 0 or text_len > clip_text_buffer.len) return -3;

    const tok_emb_idx = token_embedding_idx orelse return -4;
    const pos_emb_idx = position_embedding_idx orelse return -5;
    const ln_w_idx = ln_final_weight_idx orelse return -6;
    const ln_b_idx = ln_final_bias_idx orelse return -7;
    const proj_idx = text_projection_idx orelse return -8;

    var tokens: [CLIP_MAX_SEQ_LEN]u32 = undefined;
    const seq_len = tokenize(clip_text_buffer[0..text_len], &tokens);

    @memcpy(&debug_tokens, &tokens);
    debug_token_count = seq_len;

    for (0..seq_len) |pos| {
        const tok_id = tokens[pos];
        for (0..CLIP_EMBED_DIM) |i| {
            scratch_hidden[pos * CLIP_EMBED_DIM + i] =
                readWeight(tok_emb_idx, tok_id * CLIP_EMBED_DIM + i) +
                readWeight(pos_emb_idx, pos * CLIP_EMBED_DIM + i);
        }
    }

    @memcpy(&debug_after_embedding, scratch_hidden[0..CLIP_EMBED_DIM]);

    for (0..CLIP_NUM_LAYERS) |layer| {
        const ln1_w_idx = layer_ln1_weight_idx[layer] orelse continue;
        const ln1_b_idx = layer_ln1_bias_idx[layer] orelse continue;

        @memcpy(scratch_ln[0 .. seq_len * CLIP_EMBED_DIM], scratch_hidden[0 .. seq_len * CLIP_EMBED_DIM]);
        layerNormInPlace(&scratch_ln, seq_len, ln1_w_idx, ln1_b_idx);

        if (layer == 0) {
            @memcpy(&debug_after_ln1, scratch_ln[0..CLIP_EMBED_DIM]);
        }

        multiHeadAttention(&scratch_ln, &scratch_hidden, seq_len, layer, &scratch_q, &scratch_k, &scratch_v, &scratch_attn);

        if (layer == 0) {
            @memcpy(&debug_after_attn, scratch_hidden[0..CLIP_EMBED_DIM]);
        }

        const ln2_w_idx = layer_ln2_weight_idx[layer] orelse continue;
        const ln2_b_idx = layer_ln2_bias_idx[layer] orelse continue;

        @memcpy(scratch_ln[0 .. seq_len * CLIP_EMBED_DIM], scratch_hidden[0 .. seq_len * CLIP_EMBED_DIM]);

        if (layer == 0) {
            @memcpy(&debug_pre_ln2, scratch_ln[0..8]);
            for (0..8) |i| {
                debug_ln2_w[i] = readWeight(ln2_w_idx, i);
                debug_ln2_b[i] = readWeight(ln2_b_idx, i);
            }

            const h = scratch_ln[0..CLIP_EMBED_DIM];
            var sum: f32 = 0;
            for (h) |v| sum += v;
            debug_ln2_mean = sum / @as(f32, CLIP_EMBED_DIM);

            var var_sum: f32 = 0;
            for (h) |v| {
                const diff = v - debug_ln2_mean;
                var_sum += diff * diff;
            }
            debug_ln2_std = @sqrt(var_sum / @as(f32, CLIP_EMBED_DIM));
        }

        layerNormInPlace(&scratch_ln, seq_len, ln2_w_idx, ln2_b_idx);

        if (layer == 0) {
            @memcpy(&debug_after_ln2, scratch_ln[0..CLIP_EMBED_DIM]);
        }

        mlpBlock(&scratch_ln, &scratch_hidden, seq_len, layer);

        if (layer == 0) {
            @memcpy(&debug_after_layer0, scratch_hidden[0..CLIP_EMBED_DIM]);
        }
    }

    layerNormInPlace(&scratch_hidden, seq_len, ln_w_idx, ln_b_idx);

    const eos_pos = seq_len - 1;
    const final_hidden = scratch_hidden[eos_pos * CLIP_EMBED_DIM ..][0..CLIP_EMBED_DIM];

    for (0..CLIP_EMBED_DIM) |i| {
        var sum: f32 = 0;
        for (0..CLIP_EMBED_DIM) |j| {
            sum += final_hidden[j] * readWeight(proj_idx, i * CLIP_EMBED_DIM + j);
        }
        clip_output_buffer[i] = sum;
    }

    @memcpy(&debug_pre_norm, clip_output_buffer[0..8]);
    var pre_norm_sq: f32 = 0;
    for (clip_output_buffer) |v| pre_norm_sq += v * v;
    debug_pre_norm_norm = @sqrt(pre_norm_sq);

    var norm_sq: f32 = 0;
    for (clip_output_buffer) |v| norm_sq += v * v;
    const norm = @sqrt(norm_sq);
    if (norm > 0) {
        for (&clip_output_buffer) |*v| v.* /= norm;
    }

    return 0;
}

pub export fn clip_test_add(a: i32, b: i32) i32 {
    return a + b;
}

pub export fn clip_get_vocab_count() usize {
    return vocab_count;
}

pub export fn clip_debug_get_token(pos: usize) u32 {
    if (pos < debug_token_count) {
        return debug_tokens[pos];
    }
    return 0;
}

pub export fn clip_debug_get_token_count() usize {
    return debug_token_count;
}

pub export fn clip_debug_get_hidden(i: usize) f32 {
    if (i < CLIP_EMBED_DIM) {
        return debug_hidden[i];
    }
    return 0;
}

pub export fn clip_debug_get_token_emb(token_id: u32, dim: usize) f32 {
    const tok_emb_idx = token_embedding_idx orelse return -999;
    return readWeight(tok_emb_idx, token_id * CLIP_EMBED_DIM + dim);
}

pub export fn clip_debug_get_pos_emb(pos: usize, dim: usize) f32 {
    const pos_emb_idx = position_embedding_idx orelse return -999;
    return readWeight(pos_emb_idx, pos * CLIP_EMBED_DIM + dim);
}

pub export fn clip_debug_get_scratch_hidden(pos: usize, dim: usize) f32 {
    if (pos < CLIP_MAX_SEQ_LEN and dim < CLIP_EMBED_DIM) {
        return scratch_hidden[pos * CLIP_EMBED_DIM + dim];
    }
    return -999;
}

pub export fn clip_debug_get_after_embedding(dim: usize) f32 {
    if (dim < CLIP_EMBED_DIM) return debug_after_embedding[dim];
    return -999;
}

pub export fn clip_debug_get_after_layer0(dim: usize) f32 {
    if (dim < CLIP_EMBED_DIM) return debug_after_layer0[dim];
    return -999;
}

pub export fn clip_debug_get_after_ln1(dim: usize) f32 {
    if (dim < CLIP_EMBED_DIM) return debug_after_ln1[dim];
    return -999;
}

pub export fn clip_debug_get_after_attn(dim: usize) f32 {
    if (dim < CLIP_EMBED_DIM) return debug_after_attn[dim];
    return -999;
}

pub export fn clip_debug_get_after_ln2(dim: usize) f32 {
    if (dim < CLIP_EMBED_DIM) return debug_after_ln2[dim];
    return -999;
}

pub export fn clip_debug_get_after_fc1(dim: usize) f32 {
    if (dim < 8) return debug_after_fc1[dim];
    return -999;
}

pub export fn clip_debug_get_after_fc2(dim: usize) f32 {
    if (dim < 8) return debug_after_fc2[dim];
    return -999;
}

pub export fn clip_debug_get_ln2_w(dim: usize) f32 {
    if (dim < 8) return debug_ln2_w[dim];
    return -999;
}

pub export fn clip_debug_get_ln2_b(dim: usize) f32 {
    if (dim < 8) return debug_ln2_b[dim];
    return -999;
}

pub export fn clip_debug_get_pre_ln2(dim: usize) f32 {
    if (dim < 8) return debug_pre_ln2[dim];
    return -999;
}

pub export fn clip_debug_get_ln2_mean() f32 {
    return debug_ln2_mean;
}

pub export fn clip_debug_get_ln2_std() f32 {
    return debug_ln2_std;
}

pub export fn clip_debug_get_pre_norm(dim: usize) f32 {
    if (dim < 8) return debug_pre_norm[dim];
    return -999;
}

pub export fn clip_debug_get_pre_norm_norm() f32 {
    return debug_pre_norm_norm;
}

// ============================================================================
// Tests
// ============================================================================

test "clip_model: initialization" {
    clip_initialized = false;
    clip_model_loaded = false;
    const result = clip_init();
    try std.testing.expectEqual(@as(i32, 0), result);
    try std.testing.expect(clip_initialized);
    try std.testing.expect(!clip_model_loaded);
}

test "clip_model: buffer access" {
    const text_buf = clip_get_text_buffer();
    try std.testing.expect(text_buf != undefined);

    const text_size = clip_get_text_buffer_size();
    try std.testing.expectEqual(@as(usize, 1024), text_size);

    const out_buf = clip_get_output_buffer();
    try std.testing.expect(out_buf != undefined);

    const out_dim = clip_get_output_dim();
    try std.testing.expectEqual(@as(usize, 512), out_dim);
}

test "clip_model: weights not loaded" {
    clip_initialized = true;
    clip_model_loaded = false;
    const result = clip_weights_loaded();
    try std.testing.expectEqual(@as(i32, 0), result);
}

test "clip_model: test_add" {
    const result = clip_test_add(2, 3);
    try std.testing.expectEqual(@as(i32, 5), result);
}
