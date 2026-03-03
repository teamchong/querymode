//! EdgeQ - A Zig implementation of the Lance columnar file format reader.
//!
//! This library provides read-only access to Lance files, supporting both
//! native execution and WebAssembly (browser) targets.
//!
//! ## Example
//! ```zig
//! const edgeq = @import("edgeq");
//!
//! pub fn main() !void {
//!     var file = try edgeq.LanceFile.open("data.lance");
//!     defer file.close();
//!
//!     const footer = file.footer();
//!     std.debug.print("Columns: {}\n", .{footer.num_columns});
//! }
//! ```

const std = @import("std");

pub const format = @import("edgeq.format");
pub const io = @import("edgeq.io");
pub const proto = @import("edgeq.proto");
pub const encoding = @import("edgeq.encoding");
pub const writer = @import("edgeq.writer");
pub const table = @import("edgeq.table");
pub const dataframe = @import("edgeq.dataframe");
pub const dataset = @import("edgeq.dataset");
pub const dataset_writer = @import("edgeq.dataset_writer");

// Re-export commonly used types
pub const Footer = format.Footer;
pub const Version = format.Version;
pub const Reader = io.Reader;
pub const Table = table.Table;
pub const LanceFile = format.LanceFile;
pub const DataFrame = dataframe.DataFrame;
pub const LanceDataset = dataset.LanceDataset;
pub const LanceWriter = writer.LanceWriter;
pub const ColumnBuilder = writer.ColumnBuilder;
pub const DatasetWriter = dataset_writer.DatasetWriter;

/// Magic bytes at the end of every Lance file
pub const LANCE_MAGIC = "LANC";

/// Size of the Lance file footer in bytes
pub const FOOTER_SIZE: usize = 40;

test {
    std.testing.refAllDecls(@This());
}
