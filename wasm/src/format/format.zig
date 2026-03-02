//! File format parsing module.
//!
//! This module handles parsing columnar file formats:
//! - Lance: 40-byte footer, protobuf metadata
//! - Parquet: Thrift metadata, various encodings/compressions

const std = @import("std");

// Lance format
pub const footer = @import("footer.zig");
pub const version = @import("version.zig");
pub const lance_file = @import("lance_file.zig");
pub const lazy_lance_file = @import("lazy_lance_file.zig");
pub const manifest = @import("manifest.zig");
pub const manifest_writer = @import("manifest_writer.zig");
pub const page_row_index = @import("page_row_index.zig");

// Parquet format
pub const parquet_metadata = @import("parquet_metadata.zig");
pub const parquet_file = @import("parquet_file.zig");

// Re-export Lance types
pub const Footer = footer.Footer;
pub const Version = version.Version;
pub const LANCE_MAGIC = footer.LANCE_MAGIC;
pub const FOOTER_SIZE = footer.FOOTER_SIZE;
pub const LanceFile = lance_file.LanceFile;
pub const readFile = lance_file.readFile;
pub const LazyLanceFile = lazy_lance_file.LazyLanceFile;
pub const Manifest = manifest.Manifest;
pub const Fragment = manifest.Fragment;
pub const loadManifest = manifest.loadManifest;
pub const listVersions = manifest.listVersions;
pub const latestVersion = manifest.latestVersion;
pub const PageRowIndex = page_row_index.PageRowIndex;
pub const RowLocation = page_row_index.RowLocation;
pub const ByteRange = page_row_index.ByteRange;
pub const ManifestWriter = manifest_writer.ManifestWriter;
pub const generateUUID = manifest_writer.generateUUID;

// Re-export Parquet types
pub const ParquetFile = parquet_file.ParquetFile;
pub const ParquetError = parquet_file.ParquetError;
pub const ParquetType = parquet_metadata.Type;
pub const ParquetEncoding = parquet_metadata.Encoding;
pub const CompressionCodec = parquet_metadata.CompressionCodec;
pub const FileMetaData = parquet_metadata.FileMetaData;
pub const RowGroup = parquet_metadata.RowGroup;
pub const ColumnChunk = parquet_metadata.ColumnChunk;
pub const SchemaElement = parquet_metadata.SchemaElement;
pub const PageHeader = parquet_metadata.PageHeader;

test {
    std.testing.refAllDecls(@This());
}
