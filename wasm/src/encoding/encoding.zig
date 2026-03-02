//! Data encoding/decoding for Lance columns.
//!
//! Lance supports various encodings for column data:
//! - Plain: Direct value storage (int64, float64)
//! - Dictionary: Categorical data with lookup table
//! - RLE: Run-length encoding for repeated values
//! - UTF-8: String data with offset arrays
//!
//! Also provides parsers for common formats:
//! - CSV/TSV: Delimiter-separated values
//! - JSON/JSONL: JavaScript Object Notation
//! - Arrow IPC: Apache Arrow columnar format
//! - Avro: Apache Avro container format
//! - ORC: Optimized Row Columnar format
//! - XLSX: Microsoft Excel format

const std = @import("std");

pub const plain = @import("plain.zig");
pub const writer = @import("writer.zig");
pub const csv = @import("csv.zig");
pub const json = @import("json.zig");
pub const arrow_ipc = @import("arrow_ipc.zig");
pub const avro = @import("avro.zig");
pub const orc = @import("orc.zig");
pub const xlsx = @import("xlsx.zig");
pub const delta = @import("delta.zig");
pub const iceberg = @import("iceberg.zig");

// Re-export main types
pub const PlainDecoder = plain.PlainDecoder;
pub const PlainEncoder = writer.PlainEncoder;
pub const LanceWriter = writer.LanceWriter;
pub const FooterWriter = writer.FooterWriter;
pub const ProtobufEncoder = writer.ProtobufEncoder;
pub const DataType = writer.DataType;
pub const ColumnSchema = writer.ColumnSchema;
pub const ColumnBatch = writer.ColumnBatch;

// CSV types
pub const CsvParser = csv.CsvParser;
pub const CsvConfig = csv.Config;
pub const CsvField = csv.Field;
pub const CsvColumnType = csv.ColumnType;
pub const CsvColumnData = csv.ColumnData;
pub const readCsv = csv.readCsv;
pub const detectCsvDelimiter = csv.detectDelimiter;

// JSON types
pub const JsonFormat = json.Format;
pub const JsonConfig = json.Config;
pub const JsonColumnType = json.ColumnType;
pub const readJson = json.readJson;
pub const detectJsonFormat = json.detectFormat;

// Arrow IPC types
pub const ArrowIpcReader = arrow_ipc.ArrowIpcReader;
pub const ArrowType = arrow_ipc.ArrowType;
pub const ArrowColumnInfo = arrow_ipc.ColumnInfo;

// Avro types
pub const AvroReader = avro.AvroReader;
pub const AvroType = avro.AvroType;
pub const AvroCodec = avro.Codec;
pub const AvroFieldInfo = avro.FieldInfo;

// ORC types
pub const OrcReader = orc.OrcReader;
pub const OrcType = orc.OrcType;
pub const OrcCompressionKind = orc.CompressionKind;
pub const OrcColumnInfo = orc.ColumnInfo;

// XLSX types
pub const XlsxReader = xlsx.XlsxReader;
pub const XlsxCellType = xlsx.CellType;
pub const XlsxCellValue = xlsx.CellValue;
pub const XlsxSheet = xlsx.Sheet;

// Delta Lake types
pub const DeltaReader = delta.DeltaReader;

// Iceberg types
pub const IcebergReader = iceberg.IcebergReader;

test {
    std.testing.refAllDecls(@This());
}
