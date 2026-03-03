//! Lance file writing functionality.
//!
//! This module provides writers for creating Lance format files:
//! - LanceWriter: Write .lance data files with columnar encoding
//!
//! ## Example
//! ```zig
//! const writer = @import("querymode").writer;
//!
//! var lance_writer = writer.LanceWriter.init(allocator);
//! defer lance_writer.deinit();
//!
//! var col = try lance_writer.addColumn("id", .int64);
//! try col.appendInt64(&[_]i64{1, 2, 3});
//! try col.finalize();
//!
//! const bytes = try lance_writer.finalize();
//! ```

const std = @import("std");

pub const lance_writer = @import("lance_writer.zig");

// Re-export types
pub const LanceWriter = lance_writer.LanceWriter;
pub const ColumnBuilder = lance_writer.ColumnBuilder;
pub const ColumnType = lance_writer.ColumnType;
pub const WriteError = lance_writer.WriteError;

test {
    std.testing.refAllDecls(@This());
}
