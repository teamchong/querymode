//! Protocol encoding/decoding for Lance and Parquet metadata.
//!
//! This module provides wire format encoders and decoders:
//! - Protobuf: For Lance file metadata (read and write)
//! - Thrift TCompactProtocol: For Parquet file metadata (read-only)

const std = @import("std");

pub const decoder = @import("decoder.zig");
pub const encoder = @import("encoder.zig");
pub const lance_messages = @import("lance_messages.zig");
pub const schema = @import("schema.zig");
pub const thrift = @import("thrift.zig");

// Re-export Protobuf decoder types
pub const ProtoDecoder = decoder.ProtoDecoder;
pub const WireType = decoder.WireType;
pub const DecodeError = decoder.DecodeError;

// Re-export Protobuf encoder types
pub const ProtoEncoder = encoder.ProtoEncoder;
pub const EncodeError = encoder.EncodeError;
pub const varintSize = encoder.varintSize;
pub const signedVarintSize = encoder.signedVarintSize;

pub const ColumnMetadata = lance_messages.ColumnMetadata;
pub const Page = lance_messages.Page;
pub const Encoding = lance_messages.Encoding;

pub const Schema = schema.Schema;
pub const Field = schema.Field;
pub const FieldType = schema.FieldType;

// Re-export Thrift types
pub const ThriftDecoder = thrift.ThriftDecoder;
pub const CompactType = thrift.CompactType;
pub const ThriftError = thrift.ThriftError;

test {
    std.testing.refAllDecls(@This());
}
