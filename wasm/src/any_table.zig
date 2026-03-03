//! Unified Table Wrapper for All File Formats
//!
//! Provides a single interface to work with Lance, Parquet, Delta, Iceberg,
//! Arrow, Avro, ORC, and XLSX tables through a tagged union.

const std = @import("std");
const Table = @import("edgeq.table").Table;
const ParquetTable = @import("edgeq.parquet_table").ParquetTable;
const DeltaTable = @import("edgeq.delta_table").DeltaTable;
const IcebergTable = @import("edgeq.iceberg_table").IcebergTable;
const ArrowTable = @import("edgeq.arrow_table").ArrowTable;
const AvroTable = @import("edgeq.avro_table").AvroTable;
const OrcTable = @import("edgeq.orc_table").OrcTable;
const XlsxTable = @import("edgeq.xlsx_table").XlsxTable;

pub const AnyTable = union(enum) {
    lance: Table,
    parquet: ParquetTable,
    delta: DeltaTable,
    iceberg: IcebergTable,
    arrow: ArrowTable,
    avro: AvroTable,
    orc: OrcTable,
    xlsx: XlsxTable,

    const Self = @This();

    pub const Format = enum {
        lance,
        parquet,
        delta,
        iceberg,
        arrow,
        avro,
        orc,
        xlsx,
    };

    pub const InitInput = union(enum) {
        data: []const u8,
        path: []const u8,
    };

    pub const InitError = error{
        InvalidFormat,
        OutOfMemory,
        InvalidData,
        FileNotFound,
    };

    pub fn init(allocator: std.mem.Allocator, format: Format, input: InitInput) InitError!Self {
        return switch (format) {
            .lance => .{ .lance = Table.init(allocator, input.data) catch return InitError.InvalidData },
            .parquet => .{ .parquet = ParquetTable.init(allocator, input.data) catch return InitError.InvalidData },
            .arrow => .{ .arrow = ArrowTable.init(allocator, input.data) catch return InitError.InvalidData },
            .avro => .{ .avro = AvroTable.init(allocator, input.data) catch return InitError.InvalidData },
            .orc => .{ .orc = OrcTable.init(allocator, input.data) catch return InitError.InvalidData },
            .xlsx => .{ .xlsx = XlsxTable.init(allocator, input.data) catch return InitError.InvalidData },
            .delta => .{ .delta = DeltaTable.init(allocator, input.path) catch return InitError.InvalidData },
            .iceberg => .{ .iceberg = IcebergTable.init(allocator, input.path) catch return InitError.InvalidData },
        };
    }

    pub fn deinit(self: *Self) void {
        switch (self.*) {
            inline else => |*t| t.deinit(),
        }
    }

    pub fn numRows(self: Self) usize {
        return switch (self) {
            .lance => 0, // Lance Table doesn't have simple numRows - executor handles this
            inline else => |t| t.numRows(),
        };
    }

    pub fn getColumnNames(self: Self) [][]const u8 {
        return switch (self) {
            .lance => |t| t.columnNames() catch &[_][]const u8{},
            inline else => |t| t.getColumnNames(),
        };
    }

    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        return switch (self) {
            inline else => |t| t.columnIndex(name),
        };
    }
};
