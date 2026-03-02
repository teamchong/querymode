//! WordPiece Tokenizer for BERT-based models
//!
//! Implements WordPiece tokenization used by models like all-MiniLM-L6-v2.
//! Reference: https://huggingface.co/docs/transformers/tokenizer_summary#wordpiece

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Tokenizer = struct {
    allocator: Allocator,
    vocab: std.StringHashMap(i64),
    vocab_list: std.ArrayList([]const u8),
    unk_token_id: i64,
    cls_token_id: i64,
    sep_token_id: i64,
    pad_token_id: i64,
    max_word_chars: usize = 100,

    const Self = @This();

    // Special tokens
    pub const UNK_TOKEN = "[UNK]";
    pub const CLS_TOKEN = "[CLS]";
    pub const SEP_TOKEN = "[SEP]";
    pub const PAD_TOKEN = "[PAD]";
    pub const MASK_TOKEN = "[MASK]";

    /// Initialize tokenizer with vocabulary from JSON file
    pub fn initFromFile(allocator: Allocator, vocab_path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(vocab_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 50 * 1024 * 1024); // 50MB max
        defer allocator.free(content);

        return try initFromJson(allocator, content);
    }

    /// Initialize tokenizer with vocabulary from JSON string
    pub fn initFromJson(allocator: Allocator, json_content: []const u8) !Self {
        var vocab = std.StringHashMap(i64).init(allocator);
        errdefer vocab.deinit();

        var vocab_list = std.ArrayList([]const u8).init(allocator);
        errdefer {
            for (vocab_list.items) |item| allocator.free(item);
            vocab_list.deinit();
        }

        // Parse JSON vocabulary (format: {"token": id, ...})
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
        defer parsed.deinit();

        if (parsed.value != .object) {
            return error.InvalidVocabFormat;
        }

        // Pre-allocate vocab_list with empty strings
        const obj = parsed.value.object;
        try vocab_list.resize(obj.count());
        for (vocab_list.items) |*item| {
            item.* = "";
        }

        // Build vocabulary
        var it = obj.iterator();
        while (it.next()) |entry| {
            const token = entry.key_ptr.*;
            const id: i64 = switch (entry.value_ptr.*) {
                .integer => |i| i,
                else => continue,
            };

            // Duplicate token string
            const token_copy = try allocator.dupe(u8, token);
            try vocab.put(token_copy, id);

            // Store in list for reverse lookup
            const idx: usize = @intCast(id);
            if (idx < vocab_list.items.len) {
                vocab_list.items[idx] = token_copy;
            }
        }

        // Get special token IDs
        const unk_id = vocab.get(UNK_TOKEN) orelse 0;
        const cls_id = vocab.get(CLS_TOKEN) orelse 101;
        const sep_id = vocab.get(SEP_TOKEN) orelse 102;
        const pad_id = vocab.get(PAD_TOKEN) orelse 0;

        return Self{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_list = vocab_list,
            .unk_token_id = unk_id,
            .cls_token_id = cls_id,
            .sep_token_id = sep_id,
            .pad_token_id = pad_id,
        };
    }

    /// Initialize with a basic vocabulary (for testing)
    pub fn initBasic(allocator: Allocator) !Self {
        var vocab = std.StringHashMap(i64).init(allocator);
        const vocab_list = std.ArrayList([]const u8).init(allocator);

        // Add special tokens
        const special_tokens = [_]struct { token: []const u8, id: i64 }{
            .{ .token = PAD_TOKEN, .id = 0 },
            .{ .token = UNK_TOKEN, .id = 100 },
            .{ .token = CLS_TOKEN, .id = 101 },
            .{ .token = SEP_TOKEN, .id = 102 },
            .{ .token = MASK_TOKEN, .id = 103 },
        };

        for (special_tokens) |st| {
            const token_copy = try allocator.dupe(u8, st.token);
            try vocab.put(token_copy, st.id);
        }

        return Self{
            .allocator = allocator,
            .vocab = vocab,
            .vocab_list = vocab_list,
            .unk_token_id = 100,
            .cls_token_id = 101,
            .sep_token_id = 102,
            .pad_token_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.vocab.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.vocab.deinit();
        self.vocab_list.deinit();
    }

    /// Tokenize text and return input_ids and attention_mask
    pub fn encode(
        self: *Self,
        text: []const u8,
        max_length: usize,
    ) !struct { input_ids: []i64, attention_mask: []i64 } {
        var input_ids = try self.allocator.alloc(i64, max_length);
        errdefer self.allocator.free(input_ids);

        var attention_mask = try self.allocator.alloc(i64, max_length);
        errdefer self.allocator.free(attention_mask);

        // Initialize with padding
        @memset(input_ids, self.pad_token_id);
        @memset(attention_mask, 0);

        // Add [CLS] token
        var pos: usize = 0;
        input_ids[pos] = self.cls_token_id;
        attention_mask[pos] = 1;
        pos += 1;

        // Tokenize text
        const tokens = try self.tokenize(text);
        defer {
            for (tokens) |t| self.allocator.free(t);
            self.allocator.free(tokens);
        }

        // Add tokens (leave room for [SEP])
        for (tokens) |token| {
            if (pos >= max_length - 1) break;

            const token_id = self.vocab.get(token) orelse self.unk_token_id;
            input_ids[pos] = token_id;
            attention_mask[pos] = 1;
            pos += 1;
        }

        // Add [SEP] token
        if (pos < max_length) {
            input_ids[pos] = self.sep_token_id;
            attention_mask[pos] = 1;
        }

        return .{
            .input_ids = input_ids,
            .attention_mask = attention_mask,
        };
    }

    /// Free encoded result
    pub fn freeEncoded(self: *Self, encoded: anytype) void {
        self.allocator.free(encoded.input_ids);
        self.allocator.free(encoded.attention_mask);
    }

    /// Basic tokenization - split on whitespace and punctuation
    fn tokenize(self: *Self, text: []const u8) ![][]const u8 {
        var tokens = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (tokens.items) |t| self.allocator.free(t);
            tokens.deinit();
        }

        // Lowercase and normalize
        const lower = try self.allocator.alloc(u8, text.len);
        defer self.allocator.free(lower);
        for (text, 0..) |c, i| {
            lower[i] = std.ascii.toLower(c);
        }

        // Split into words
        var words = std.ArrayList([]const u8).init(self.allocator);
        defer words.deinit();

        var word_start: ?usize = null;
        for (lower, 0..) |c, i| {
            const is_word_char = std.ascii.isAlphanumeric(c);

            if (is_word_char) {
                if (word_start == null) {
                    word_start = i;
                }
            } else {
                if (word_start) |start| {
                    try words.append(lower[start..i]);
                    word_start = null;
                }
                if (std.ascii.isPunct(c)) {
                    const punct = try self.allocator.dupe(u8, &[_]u8{c});
                    try tokens.append(punct);
                }
            }
        }
        // Handle last word
        if (word_start) |start| {
            try words.append(lower[start..]);
        }

        // Apply WordPiece to each word
        for (words.items) |word| {
            const word_tokens = try self.wordPieceTokenize(word);
            defer self.allocator.free(word_tokens);
            for (word_tokens) |wt| {
                try tokens.append(wt);
            }
        }

        return try tokens.toOwnedSlice();
    }

    /// WordPiece tokenization for a single word
    fn wordPieceTokenize(self: *Self, word: []const u8) ![][]const u8 {
        if (word.len == 0) return try self.allocator.alloc([]const u8, 0);
        if (word.len > self.max_word_chars) {
            // Word too long, return [UNK]
            var result = try self.allocator.alloc([]const u8, 1);
            result[0] = try self.allocator.dupe(u8, UNK_TOKEN);
            return result;
        }

        var tokens = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (tokens.items) |t| self.allocator.free(t);
            tokens.deinit();
        }

        var start: usize = 0;
        while (start < word.len) {
            var end = word.len;
            var cur_substr: ?[]const u8 = null;

            while (start < end) {
                var substr: []const u8 = undefined;
                if (start > 0) {
                    // Add ## prefix for continuation
                    const sub_with_prefix = try std.fmt.allocPrint(self.allocator, "##{s}", .{word[start..end]});
                    defer if (cur_substr == null) self.allocator.free(sub_with_prefix);
                    substr = sub_with_prefix;
                } else {
                    substr = word[start..end];
                }

                if (self.vocab.contains(substr)) {
                    cur_substr = try self.allocator.dupe(u8, substr);
                    if (start > 0) self.allocator.free(substr);
                    break;
                }

                if (start > 0) self.allocator.free(substr);
                end -= 1;
            }

            if (cur_substr) |s| {
                try tokens.append(s);
                start = end;
            } else {
                // Unknown token
                try tokens.append(try self.allocator.dupe(u8, UNK_TOKEN));
                start += 1;
            }
        }

        return try tokens.toOwnedSlice();
    }
};

// =============================================================================
// Tests
// =============================================================================

test "basic tokenizer" {
    const allocator = std.testing.allocator;
    var tokenizer = try Tokenizer.initBasic(allocator);
    defer tokenizer.deinit();

    // Test encoding
    const encoded = try tokenizer.encode("hello world", 32);
    defer tokenizer.freeEncoded(encoded);

    // First token should be [CLS]
    try std.testing.expectEqual(@as(i64, 101), encoded.input_ids[0]);
    try std.testing.expectEqual(@as(i64, 1), encoded.attention_mask[0]);
}
