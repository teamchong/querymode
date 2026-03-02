//! GGUF Model Format Parser
//!
//! Native implementation for loading GGUF model files.
//! Supports F16/F32 tensors and WordPiece vocabulary.

const std = @import("std");

// ============================================================================
// GGUF Type Constants
// ============================================================================

pub const TYPE_STRING: u32 = 8;
pub const TYPE_ARRAY: u32 = 9;
pub const TYPE_UINT32: u32 = 4;
pub const TYPE_FLOAT32: u32 = 6;
pub const TYPE_BOOL: u32 = 7;
pub const TYPE_UINT64: u32 = 10;

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;

// ============================================================================
// GGUF Header
// ============================================================================

pub const Header = struct {
    version: u32,
    n_tensors: u64,
    n_kv: u64,
};

// ============================================================================
// Tensor Info
// ============================================================================

pub const TensorInfo = struct {
    name: []const u8,
    offset: u64,
    dtype: u32,
    dims: [4]u64,
    n_dims: u32,

    pub fn numElements(self: TensorInfo) usize {
        var n: usize = 1;
        for (0..self.n_dims) |i| {
            n *= @intCast(self.dims[i]);
        }
        return n;
    }

    pub fn byteSize(self: TensorInfo) usize {
        const n = self.numElements();
        return switch (self.dtype) {
            GGML_TYPE_F32 => n * 4,
            GGML_TYPE_F16 => n * 2,
            else => n * 4,
        };
    }
};

// ============================================================================
// GGUF Model
// ============================================================================

pub const Model = struct {
    allocator: std.mem.Allocator,
    data: []align(32) const u8,
    header: Header,
    tensors: []TensorInfo,
    tensor_map: std.StringHashMap(usize),
    data_offset: usize,

    // Vocabulary
    vocab_tokens: [][]const u8,
    vocab_count: usize,

    pub fn init(allocator: std.mem.Allocator) Model {
        return .{
            .allocator = allocator,
            .data = &[_]u8{},
            .header = .{ .version = 0, .n_tensors = 0, .n_kv = 0 },
            .tensors = &[_]TensorInfo{},
            .tensor_map = std.StringHashMap(usize).init(allocator),
            .data_offset = 0,
            .vocab_tokens = &[_][]const u8{},
            .vocab_count = 0,
        };
    }

    pub fn deinit(self: *Model) void {
        self.tensor_map.deinit();
        if (self.tensors.len > 0) {
            for (self.tensors) |t| {
                self.allocator.free(t.name);
            }
            self.allocator.free(self.tensors);
        }
        if (self.vocab_tokens.len > 0) {
            for (self.vocab_tokens[0..self.vocab_count]) |tok| {
                self.allocator.free(tok);
            }
            self.allocator.free(self.vocab_tokens);
        }
        if (self.data.len > 0) {
            self.allocator.free(self.data);
        }
    }

    pub fn loadFromFile(self: *Model, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const size = stat.size;

        const data = try self.allocator.alignedAlloc(u8, .@"32", size);
        errdefer self.allocator.free(data);

        const bytes_read = try file.readAll(data);
        if (bytes_read != size) return error.IncompleteRead;

        try self.loadFromMemory(data);
    }

    pub fn loadFromMemory(self: *Model, data: []align(32) const u8) !void {
        if (data.len < 24) return error.FileTooSmall;

        // Verify magic
        if (!std.mem.eql(u8, data[0..4], "GGUF")) return error.InvalidMagic;

        self.data = data;
        self.header.version = readU32LE(data, 4);
        if (self.header.version < 2 or self.header.version > 3) return error.UnsupportedVersion;

        self.header.n_tensors = readU64LE(data, 8);
        self.header.n_kv = readU64LE(data, 16);

        var pos: usize = 24;

        // Parse key-value metadata
        for (0..self.header.n_kv) |_| {
            const key = readString(data, &pos);
            if (pos + 4 > data.len) return error.UnexpectedEof;
            const vtype = readU32LE(data, pos);
            pos += 4;

            if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
                if (vtype != TYPE_ARRAY) {
                    skipValue(data, &pos, vtype);
                    continue;
                }

                if (pos + 12 > data.len) return error.UnexpectedEof;
                const atype = readU32LE(data, pos);
                pos += 4;
                const alen: usize = @intCast(readU64LE(data, pos));
                pos += 8;

                if (atype == TYPE_STRING) {
                    self.vocab_tokens = try self.allocator.alloc([]const u8, alen);
                    errdefer self.allocator.free(self.vocab_tokens);

                    for (0..alen) |i| {
                        const tok = readString(data, &pos);
                        self.vocab_tokens[i] = try self.allocator.dupe(u8, tok);
                    }
                    self.vocab_count = alen;
                } else {
                    for (0..alen) |_| {
                        skipValue(data, &pos, atype);
                    }
                }
            } else {
                skipValue(data, &pos, vtype);
            }
        }

        // Parse tensor info
        const n_tensors: usize = @intCast(self.header.n_tensors);
        self.tensors = try self.allocator.alloc(TensorInfo, n_tensors);
        errdefer self.allocator.free(self.tensors);

        for (0..n_tensors) |i| {
            const name = readString(data, &pos);
            self.tensors[i].name = try self.allocator.dupe(u8, name);

            if (pos + 4 > data.len) return error.UnexpectedEof;
            const n_dims = readU32LE(data, pos);
            pos += 4;
            self.tensors[i].n_dims = n_dims;

            self.tensors[i].dims = .{ 1, 1, 1, 1 };
            for (0..@min(n_dims, 4)) |d| {
                if (pos + 8 > data.len) return error.UnexpectedEof;
                self.tensors[i].dims[d] = readU64LE(data, pos);
                pos += 8;
            }

            if (pos + 12 > data.len) return error.UnexpectedEof;
            self.tensors[i].dtype = readU32LE(data, pos);
            pos += 4;
            self.tensors[i].offset = readU64LE(data, pos);
            pos += 8;

            try self.tensor_map.put(self.tensors[i].name, i);
        }

        // Data starts at 32-byte aligned offset
        self.data_offset = (pos + 31) & ~@as(usize, 31);
    }

    pub fn findTensor(self: *const Model, name: []const u8) ?*const TensorInfo {
        const idx = self.tensor_map.get(name) orelse return null;
        return &self.tensors[idx];
    }

    pub fn getTensorData(self: *const Model, tensor: *const TensorInfo) []const u8 {
        const start = self.data_offset + @as(usize, @intCast(tensor.offset));
        const end = start + tensor.byteSize();
        return self.data[start..end];
    }

    pub fn readWeightF32(self: *const Model, tensor: *const TensorInfo, idx: usize) f32 {
        const data = self.getTensorData(tensor);
        if (tensor.dtype == GGML_TYPE_F16) {
            const ptr: [*]const u16 = @ptrCast(@alignCast(data.ptr));
            return f16ToF32(ptr[idx]);
        } else {
            const ptr: [*]const f32 = @ptrCast(@alignCast(data.ptr));
            return ptr[idx];
        }
    }

    pub fn readWeightRowF32(self: *const Model, tensor: *const TensorInfo, row: usize, row_len: usize, buf: []f32) void {
        const data = self.getTensorData(tensor);
        const row_offset = row * row_len;

        if (tensor.dtype == GGML_TYPE_F16) {
            const ptr: [*]const u16 = @ptrCast(@alignCast(data.ptr));
            for (0..row_len) |i| {
                buf[i] = f16ToF32(ptr[row_offset + i]);
            }
        } else {
            const ptr: [*]const f32 = @ptrCast(@alignCast(data.ptr));
            @memcpy(buf[0..row_len], ptr[row_offset .. row_offset + row_len]);
        }
    }

    pub fn getVocabToken(self: *const Model, idx: usize) []const u8 {
        if (idx >= self.vocab_count) return "";
        return self.vocab_tokens[idx];
    }
};

// ============================================================================
// Binary Reading Helpers
// ============================================================================

fn readU32LE(data: []const u8, offset: usize) u32 {
    return @as(u32, data[offset]) |
        (@as(u32, data[offset + 1]) << 8) |
        (@as(u32, data[offset + 2]) << 16) |
        (@as(u32, data[offset + 3]) << 24);
}

fn readU64LE(data: []const u8, offset: usize) u64 {
    return @as(u64, data[offset]) |
        (@as(u64, data[offset + 1]) << 8) |
        (@as(u64, data[offset + 2]) << 16) |
        (@as(u64, data[offset + 3]) << 24) |
        (@as(u64, data[offset + 4]) << 32) |
        (@as(u64, data[offset + 5]) << 40) |
        (@as(u64, data[offset + 6]) << 48) |
        (@as(u64, data[offset + 7]) << 56);
}

fn readString(data: []const u8, pos: *usize) []const u8 {
    if (pos.* + 8 > data.len) return "";
    const len: usize = @intCast(readU64LE(data, pos.*));
    pos.* += 8;
    if (pos.* + len > data.len) return "";
    const str = data[pos.* .. pos.* + len];
    pos.* += len;
    return str;
}

fn skipValue(data: []const u8, pos: *usize, vtype: u32) void {
    switch (vtype) {
        TYPE_STRING => {
            _ = readString(data, pos);
        },
        TYPE_UINT32, TYPE_FLOAT32 => {
            pos.* += 4;
        },
        TYPE_BOOL => {
            pos.* += 1;
        },
        TYPE_UINT64 => {
            pos.* += 8;
        },
        TYPE_ARRAY => {
            if (pos.* + 12 > data.len) return;
            const atype = readU32LE(data, pos.*);
            pos.* += 4;
            const alen: usize = @intCast(readU64LE(data, pos.*));
            pos.* += 8;
            for (0..alen) |_| {
                skipValue(data, pos, atype);
            }
        },
        else => {},
    }
}

// ============================================================================
// Float16 Conversion
// ============================================================================

pub fn f16ToF32(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15) << 31;
    const exp: u32 = (h >> 10) & 0x1F;
    const mant: u32 = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            return @bitCast(sign);
        }
        // Subnormal
        var e: u32 = 0;
        var m = mant;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e += 1;
        }
        const new_exp = (127 - 15 - e) << 23;
        const new_mant = (m & 0x3FF) << 13;
        return @bitCast(sign | new_exp | new_mant);
    } else if (exp == 31) {
        // Inf/NaN
        return @bitCast(sign | 0x7F800000 | (mant << 13));
    }

    const new_exp = (exp + 127 - 15) << 23;
    const new_mant = mant << 13;
    return @bitCast(sign | new_exp | new_mant);
}

// ============================================================================
// Math Functions
// ============================================================================

/// Error function (erf) approximation using Abramowitz and Stegun formula 7.1.26
pub fn erf(x: f32) f32 {
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;

    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const abs_x = @abs(x);

    const t = 1.0 / (1.0 + p * abs_x);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;

    const y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * @exp(-abs_x * abs_x);

    return sign * y;
}

/// Standard GELU activation
pub fn gelu(x: f32) f32 {
    const sqrt2_inv: f32 = 0.7071067811865476;
    return x * 0.5 * (1.0 + erf(x * sqrt2_inv));
}

// ============================================================================
// Tests
// ============================================================================

test "gguf: f16ToF32 conversions" {
    try std.testing.expectEqual(@as(f32, 0.0), f16ToF32(0));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), f16ToF32(0x3C00), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), f16ToF32(0xBC00), 0.0001);
}

test "gguf: erf function" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), erf(0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8427), erf(1.0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.8427), erf(-1.0), 0.001);
}

test "gguf: gelu function" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu(0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413), gelu(1.0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1587), gelu(-1.0), 0.001);
}
