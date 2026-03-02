//! ORC Stream and Stripe Metadata Parsing
//!
//! ORC files are organized into stripes. Each stripe contains:
//! - Index data (row group indices)
//! - Data streams (actual column data)
//! - Stripe footer (protobuf metadata)
//!
//! The stripe footer contains:
//! - Stream descriptors (kind, column, length)
//! - Column encodings (DIRECT, DICTIONARY, etc.)
//!
//! Reference: https://orc.apache.org/specification/ORCv1/

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Stream kinds in ORC
pub const StreamKind = enum(u8) {
    present = 0, // Null bitmap (RLE boolean)
    data = 1, // Primary data stream
    length = 2, // String/binary lengths
    dictionary_data = 3, // Dictionary values
    dictionary_count = 4, // Number of dictionary entries (deprecated)
    secondary = 5, // Secondary data (e.g., nanoseconds)
    row_index = 6, // Row group index
    bloom_filter = 7, // Bloom filter
    bloom_filter_utf8 = 8, // UTF8 bloom filter
    encrypted_index = 9,
    encrypted_data = 10,
    stripe_statistics = 11,
    file_statistics = 12,
    _,

    pub fn fromInt(value: u64) StreamKind {
        return @enumFromInt(@as(u8, @truncate(value)));
    }
};

/// Column encoding kinds
pub const ColumnEncodingKind = enum(u8) {
    direct = 0,
    dictionary = 1,
    direct_v2 = 2,
    dictionary_v2 = 3,
    _,

    pub fn fromInt(value: u64) ColumnEncodingKind {
        return @enumFromInt(@as(u8, @truncate(value)));
    }

    pub fn usesRleV2(self: ColumnEncodingKind) bool {
        return self == .direct_v2 or self == .dictionary_v2;
    }

    pub fn usesDictionary(self: ColumnEncodingKind) bool {
        return self == .dictionary or self == .dictionary_v2;
    }
};

/// Stream descriptor from StripeFooter
pub const StreamInfo = struct {
    kind: StreamKind,
    column_id: u32,
    length: u64,
    offset: u64 = 0, // Computed after parsing
};

/// Column encoding from StripeFooter
pub const ColumnEncoding = struct {
    kind: ColumnEncodingKind,
    dictionary_size: u32 = 0,
};

/// Stripe information from file footer
pub const StripeInfo = struct {
    offset: u64,
    index_length: u64,
    data_length: u64,
    footer_length: u64,
    number_of_rows: u64,
};

/// Parsed stripe footer
pub const StripeFooter = struct {
    streams: []StreamInfo,
    columns: []ColumnEncoding,
    allocator: Allocator,

    pub fn deinit(self: *StripeFooter) void {
        if (self.streams.len > 0) {
            self.allocator.free(self.streams);
        }
        if (self.columns.len > 0) {
            self.allocator.free(self.columns);
        }
    }

    /// Find a stream by column ID and kind
    pub fn findStream(self: StripeFooter, column_id: u32, kind: StreamKind) ?StreamInfo {
        for (self.streams) |stream| {
            if (stream.column_id == column_id and stream.kind == kind) {
                return stream;
            }
        }
        return null;
    }
};

/// Parse errors
pub const ParseError = error{
    InvalidProtobuf,
    UnexpectedEof,
    OutOfMemory,
};

/// Parse stripe footer protobuf
pub fn parseStripeFooter(data: []const u8, allocator: Allocator) ParseError!StripeFooter {
    var streams = std.ArrayListUnmanaged(StreamInfo){};
    errdefer streams.deinit(allocator);

    var columns = std.ArrayListUnmanaged(ColumnEncoding){};
    errdefer columns.deinit(allocator);

    var pos: usize = 0;

    while (pos < data.len) {
        const tag = readVarint(data, &pos) catch return ParseError.InvalidProtobuf;
        const field_num = tag >> 3;
        const wire_type: u3 = @truncate(tag);

        switch (field_num) {
            1 => {
                // streams (repeated Stream message)
                if (wire_type != 2) return ParseError.InvalidProtobuf;
                const len = readVarint(data, &pos) catch return ParseError.InvalidProtobuf;
                if (pos + len > data.len) return ParseError.UnexpectedEof;

                const stream = parseStream(data[pos..][0..len]) catch return ParseError.InvalidProtobuf;
                streams.append(allocator, stream) catch return ParseError.OutOfMemory;
                pos += len;
            },
            2 => {
                // columns (repeated ColumnEncoding message)
                if (wire_type != 2) return ParseError.InvalidProtobuf;
                const len = readVarint(data, &pos) catch return ParseError.InvalidProtobuf;
                if (pos + len > data.len) return ParseError.UnexpectedEof;

                const encoding = parseColumnEncoding(data[pos..][0..len]) catch return ParseError.InvalidProtobuf;
                columns.append(allocator, encoding) catch return ParseError.OutOfMemory;
                pos += len;
            },
            else => {
                // Skip unknown fields
                pos = skipField(data, pos, wire_type) catch return ParseError.InvalidProtobuf;
            },
        }
    }

    // Compute stream offsets separately for index and data sections
    // Index streams (ROW_INDEX, BLOOM_FILTER_UTF8) go in index section
    // Data streams (PRESENT, DATA, LENGTH, etc.) go in data section
    var index_offset: u64 = 0;
    var data_offset: u64 = 0;
    for (streams.items) |*stream| {
        if (stream.kind == .row_index or stream.kind == .bloom_filter or stream.kind == .bloom_filter_utf8) {
            stream.offset = index_offset;
            index_offset += stream.length;
        } else {
            stream.offset = data_offset;
            data_offset += stream.length;
        }
    }

    return StripeFooter{
        .streams = streams.toOwnedSlice(allocator) catch return ParseError.OutOfMemory,
        .columns = columns.toOwnedSlice(allocator) catch return ParseError.OutOfMemory,
        .allocator = allocator,
    };
}

/// Parse a single Stream message
fn parseStream(data: []const u8) !StreamInfo {
    var stream = StreamInfo{
        .kind = .data,
        .column_id = 0,
        .length = 0,
    };

    var pos: usize = 0;
    while (pos < data.len) {
        const tag = try readVarint(data, &pos);
        const field_num = tag >> 3;
        const wire_type: u3 = @truncate(tag);

        switch (field_num) {
            1 => {
                // kind (enum)
                if (wire_type != 0) return error.InvalidProtobuf;
                const value = try readVarint(data, &pos);
                stream.kind = StreamKind.fromInt(value);
            },
            2 => {
                // column (uint32)
                if (wire_type != 0) return error.InvalidProtobuf;
                stream.column_id = @truncate(try readVarint(data, &pos));
            },
            3 => {
                // length (uint64)
                if (wire_type != 0) return error.InvalidProtobuf;
                stream.length = try readVarint(data, &pos);
            },
            else => {
                pos = try skipField(data, pos, wire_type);
            },
        }
    }

    return stream;
}

/// Parse a single ColumnEncoding message
fn parseColumnEncoding(data: []const u8) !ColumnEncoding {
    var encoding = ColumnEncoding{
        .kind = .direct,
        .dictionary_size = 0,
    };

    var pos: usize = 0;
    while (pos < data.len) {
        const tag = try readVarint(data, &pos);
        const field_num = tag >> 3;
        const wire_type: u3 = @truncate(tag);

        switch (field_num) {
            1 => {
                // kind (enum)
                if (wire_type != 0) return error.InvalidProtobuf;
                const value = try readVarint(data, &pos);
                encoding.kind = ColumnEncodingKind.fromInt(value);
            },
            2 => {
                // dictionarySize (uint32)
                if (wire_type != 0) return error.InvalidProtobuf;
                encoding.dictionary_size = @truncate(try readVarint(data, &pos));
            },
            else => {
                pos = try skipField(data, pos, wire_type);
            },
        }
    }

    return encoding;
}

/// Parse stripe information from file footer
pub fn parseStripeInfo(data: []const u8) !StripeInfo {
    var stripe = StripeInfo{
        .offset = 0,
        .index_length = 0,
        .data_length = 0,
        .footer_length = 0,
        .number_of_rows = 0,
    };

    var pos: usize = 0;
    while (pos < data.len) {
        const tag = try readVarint(data, &pos);
        const field_num = tag >> 3;
        const wire_type: u3 = @truncate(tag);

        switch (field_num) {
            1 => {
                // offset (uint64)
                if (wire_type != 0) return error.InvalidProtobuf;
                stripe.offset = try readVarint(data, &pos);
            },
            2 => {
                // indexLength (uint64)
                if (wire_type != 0) return error.InvalidProtobuf;
                stripe.index_length = try readVarint(data, &pos);
            },
            3 => {
                // dataLength (uint64)
                if (wire_type != 0) return error.InvalidProtobuf;
                stripe.data_length = try readVarint(data, &pos);
            },
            4 => {
                // footerLength (uint64)
                if (wire_type != 0) return error.InvalidProtobuf;
                stripe.footer_length = try readVarint(data, &pos);
            },
            5 => {
                // numberOfRows (uint64)
                if (wire_type != 0) return error.InvalidProtobuf;
                stripe.number_of_rows = try readVarint(data, &pos);
            },
            else => {
                pos = try skipField(data, pos, wire_type);
            },
        }
    }

    return stripe;
}

// ============================================================================
// Protobuf helpers
// ============================================================================

/// Read unsigned varint
fn readVarint(data: []const u8, pos: *usize) !u64 {
    var result: u64 = 0;
    var shift: u6 = 0;

    while (pos.* < data.len) {
        const byte = data[pos.*];
        pos.* += 1;

        result |= @as(u64, byte & 0x7F) << shift;

        if (byte & 0x80 == 0) {
            return result;
        }

        shift +|= 7;
        if (shift > 63) return error.InvalidProtobuf;
    }

    return error.UnexpectedEof;
}

/// Skip a protobuf field based on wire type
fn skipField(data: []const u8, start: usize, wire_type: u3) !usize {
    var pos = start;

    switch (wire_type) {
        0 => {
            // Varint - skip bytes until high bit is 0
            while (pos < data.len and data[pos] & 0x80 != 0) {
                pos += 1;
            }
            if (pos < data.len) pos += 1;
        },
        1 => {
            // 64-bit fixed
            pos += 8;
        },
        2 => {
            // Length-delimited
            const len = try readVarint(data, &pos);
            pos += len;
        },
        5 => {
            // 32-bit fixed
            pos += 4;
        },
        else => {
            return error.InvalidProtobuf;
        },
    }

    return pos;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "stream kind from int" {
    try testing.expectEqual(StreamKind.present, StreamKind.fromInt(0));
    try testing.expectEqual(StreamKind.data, StreamKind.fromInt(1));
    try testing.expectEqual(StreamKind.length, StreamKind.fromInt(2));
    try testing.expectEqual(StreamKind.dictionary_data, StreamKind.fromInt(3));
}

test "column encoding kind" {
    try testing.expect(ColumnEncodingKind.direct_v2.usesRleV2());
    try testing.expect(ColumnEncodingKind.dictionary_v2.usesRleV2());
    try testing.expect(!ColumnEncodingKind.direct.usesRleV2());

    try testing.expect(ColumnEncodingKind.dictionary.usesDictionary());
    try testing.expect(ColumnEncodingKind.dictionary_v2.usesDictionary());
    try testing.expect(!ColumnEncodingKind.direct.usesDictionary());
}
