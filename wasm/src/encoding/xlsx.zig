//! Microsoft Excel XLSX File Reader
//!
//! Reads Excel files (.xlsx) which are ZIP archives containing:
//! - xl/worksheets/sheet1.xml - sheet data
//! - xl/sharedStrings.xml - shared string table (optional)
//! - xl/workbook.xml - workbook metadata
//!
//! Reference: http://www.ecma-international.org/publications/standards/Ecma-376.htm

const std = @import("std");
const Allocator = std.mem.Allocator;

/// ZIP magic bytes
const ZIP_LOCAL_FILE_MAGIC: u32 = 0x04034b50; // "PK\x03\x04"
const ZIP_CENTRAL_DIR_MAGIC: u32 = 0x02014b50; // "PK\x01\x02"
const ZIP_END_CENTRAL_DIR_MAGIC: u32 = 0x06054b50; // "PK\x05\x06"

/// Compression methods
const COMPRESSION_STORED: u16 = 0;
const COMPRESSION_DEFLATE: u16 = 8;

/// Cell types
pub const CellType = enum {
    number,
    string,
    boolean,
    error_type,
    shared_string,
    inline_string,
    formula,
    empty,
};

/// Cell value
pub const CellValue = union(CellType) {
    number: f64,
    string: []const u8,
    boolean: bool,
    error_type: []const u8,
    shared_string: usize,
    inline_string: []const u8,
    formula: []const u8,
    empty: void,
};

/// Sheet data
pub const Sheet = struct {
    name: []const u8,
    rows: usize,
    cols: usize,
};

/// Result of extracting a file from ZIP
const ExtractResult = struct {
    data: []const u8,
    allocated: bool,
};

/// XLSX file reader
pub const XlsxReader = struct {
    allocator: Allocator,
    data: []const u8,

    // Extracted content
    sheets: []Sheet,
    num_rows: usize,
    num_cols: usize,

    // Cell data (simplified: just first sheet, first 100 rows)
    cell_values: [][]CellValue,

    const Self = @This();

    /// Initialize reader from file data
    pub fn init(allocator: Allocator, data: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .data = data,
            .sheets = &.{},
            .num_rows = 0,
            .num_cols = 0,
            .cell_values = &.{},
        };

        try self.parseFile();
        return self;
    }

    pub fn deinit(self: *Self) void {
        // Free cell values
        for (self.cell_values) |row| {
            for (row) |cell| {
                switch (cell) {
                    .string => |s| self.allocator.free(s),
                    .inline_string => |s| self.allocator.free(s),
                    .error_type => |s| self.allocator.free(s),
                    .formula => |s| self.allocator.free(s),
                    else => {},
                }
            }
            self.allocator.free(row);
        }
        if (self.cell_values.len > 0) {
            self.allocator.free(self.cell_values);
        }

        // Free sheets
        for (self.sheets) |sheet| {
            self.allocator.free(sheet.name);
        }
        if (self.sheets.len > 0) {
            self.allocator.free(self.sheets);
        }
    }

    /// Parse XLSX file
    fn parseFile(self: *Self) !void {
        // Validate ZIP magic
        if (self.data.len < 4) {
            return error.InvalidXlsxFile;
        }

        const magic = std.mem.readInt(u32, self.data[0..4], .little);
        if (magic != ZIP_LOCAL_FILE_MAGIC) {
            return error.InvalidZipMagic;
        }

        // Find and extract sheet1.xml
        const sheet_data = try self.extractFile("xl/worksheets/sheet1.xml");
        defer if (sheet_data.allocated) self.allocator.free(sheet_data.data);

        // Parse sheet XML
        try self.parseSheetXml(sheet_data.data);
    }

    /// Extract a file from the ZIP archive
    fn extractFile(self: *Self, filename: []const u8) !ExtractResult {
        var pos: usize = 0;

        while (pos + 30 < self.data.len) {
            const sig = std.mem.readInt(u32, self.data[pos..][0..4], .little);

            if (sig != ZIP_LOCAL_FILE_MAGIC) {
                // End of local files
                break;
            }

            // Parse local file header
            const compression = std.mem.readInt(u16, self.data[pos + 8 ..][0..2], .little);
            const compressed_size = std.mem.readInt(u32, self.data[pos + 18 ..][0..4], .little);
            const uncompressed_size = std.mem.readInt(u32, self.data[pos + 22 ..][0..4], .little);
            const name_len = std.mem.readInt(u16, self.data[pos + 26 ..][0..2], .little);
            const extra_len = std.mem.readInt(u16, self.data[pos + 28 ..][0..2], .little);

            const name_start = pos + 30;
            const name_end = name_start + name_len;
            const data_start = name_end + extra_len;

            if (name_end > self.data.len) break;

            const name = self.data[name_start..name_end];

            if (std.mem.eql(u8, name, filename)) {
                // Found the file
                const data_end = data_start + compressed_size;
                if (data_end > self.data.len) {
                    return error.InvalidZipFile;
                }

                const file_data = self.data[data_start..data_end];

                if (compression == COMPRESSION_STORED) {
                    // Uncompressed
                    return ExtractResult{ .data = file_data, .allocated = false };
                } else if (compression == COMPRESSION_DEFLATE) {
                    // Need to decompress
                    return try self.decompressDeflate(file_data, uncompressed_size);
                } else {
                    return error.UnsupportedCompression;
                }
            }

            // Move to next file
            pos = data_start + compressed_size;
        }

        return error.FileNotFound;
    }

    /// Decompress deflate-compressed data
    fn decompressDeflate(self: *Self, compressed: []const u8, uncompressed_size: usize) !ExtractResult {
        // Create input reader from compressed data
        var input_reader = std.Io.Reader.fixed(compressed);

        // Create window buffer for decompression
        var window_buf: [std.compress.flate.max_window_len]u8 = undefined;

        // Initialize decompressor for raw deflate (no zlib headers - ZIP uses raw deflate)
        var decomp = std.compress.flate.Decompress.init(&input_reader, .raw, &window_buf);

        // Allocate output buffer with expected uncompressed size
        const result = decomp.reader.allocRemaining(self.allocator, std.Io.Limit.limited(uncompressed_size * 2)) catch |err| {
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.StreamTooLong => error.InvalidXlsxFile,
                else => error.InvalidXlsxFile,
            };
        };

        // Check for decompression errors
        if (decomp.err) |_| {
            self.allocator.free(result);
            return error.InvalidXlsxFile;
        }

        return .{ .data = result, .allocated = true };
    }

    /// Parse sheet XML to extract cell data
    fn parseSheetXml(self: *Self, xml: []const u8) !void {
        var rows = std.ArrayList([]CellValue){};
        errdefer {
            for (rows.items) |row| {
                for (row) |cell| {
                    switch (cell) {
                        .string => |s| self.allocator.free(s),
                        .inline_string => |s| self.allocator.free(s),
                        .error_type => |s| self.allocator.free(s),
                        .formula => |s| self.allocator.free(s),
                        else => {},
                    }
                }
                self.allocator.free(row);
            }
            rows.deinit(self.allocator);
        }

        var max_cols: usize = 0;

        // Simple XML parsing - find <row> elements
        var pos: usize = 0;
        while (pos < xml.len) {
            // Find next <row
            const row_start = std.mem.indexOfPos(u8, xml, pos, "<row") orelse break;
            const row_end = std.mem.indexOfPos(u8, xml, row_start, "</row>") orelse break;

            const row_xml = xml[row_start .. row_end + 6];

            // Parse cells in this row
            var cells = std.ArrayList(CellValue){};
            errdefer {
                for (cells.items) |cell| {
                    switch (cell) {
                        .string => |s| self.allocator.free(s),
                        .inline_string => |s| self.allocator.free(s),
                        .error_type => |s| self.allocator.free(s),
                        .formula => |s| self.allocator.free(s),
                        else => {},
                    }
                }
                cells.deinit(self.allocator);
            }

            var cell_pos: usize = 0;
            while (cell_pos < row_xml.len) {
                // Find next <c (cell)
                const c_start = std.mem.indexOfPos(u8, row_xml, cell_pos, "<c ") orelse break;
                const c_end = std.mem.indexOfPos(u8, row_xml, c_start, "</c>") orelse
                    std.mem.indexOfPos(u8, row_xml, c_start, "/>") orelse break;

                const cell_xml = row_xml[c_start..@min(c_end + 4, row_xml.len)];

                // Parse cell
                const cell_value = try self.parseCell(cell_xml);
                try cells.append(self.allocator, cell_value);

                cell_pos = c_end + 1;
            }

            if (cells.items.len > 0) {
                max_cols = @max(max_cols, cells.items.len);
                try rows.append(self.allocator, try cells.toOwnedSlice(self.allocator));
            } else {
                cells.deinit(self.allocator);
            }

            pos = row_end + 6;
        }

        self.cell_values = try rows.toOwnedSlice(self.allocator);
        self.num_rows = self.cell_values.len;
        self.num_cols = max_cols;
    }

    /// Parse a single cell
    fn parseCell(self: *Self, cell_xml: []const u8) !CellValue {
        // Check cell type attribute t="..."
        const type_attr = findAttr(cell_xml, "t=");

        // Look for value in <v>...</v>
        if (std.mem.indexOf(u8, cell_xml, "<v>")) |v_start| {
            if (std.mem.indexOfPos(u8, cell_xml, v_start, "</v>")) |v_end| {
                const value_str = cell_xml[v_start + 3 .. v_end];

                if (type_attr) |t| {
                    if (std.mem.eql(u8, t, "n")) {
                        // Number
                        const num = std.fmt.parseFloat(f64, value_str) catch 0;
                        return .{ .number = num };
                    } else if (std.mem.eql(u8, t, "s")) {
                        // Shared string index
                        const idx = std.fmt.parseInt(usize, value_str, 10) catch 0;
                        return .{ .shared_string = idx };
                    } else if (std.mem.eql(u8, t, "b")) {
                        // Boolean
                        return .{ .boolean = std.mem.eql(u8, value_str, "1") };
                    }
                }

                // Default to number
                const num = std.fmt.parseFloat(f64, value_str) catch 0;
                return .{ .number = num };
            }
        }

        // Look for inline string <is><t>...</t></is>
        if (std.mem.indexOf(u8, cell_xml, "<t>")) |t_start| {
            if (std.mem.indexOfPos(u8, cell_xml, t_start, "</t>")) |t_end| {
                const text = cell_xml[t_start + 3 .. t_end];
                const text_copy = try self.allocator.dupe(u8, text);
                return .{ .inline_string = text_copy };
            }
        }

        return .{ .empty = {} };
    }

    /// Find attribute value in XML
    fn findAttr(xml: []const u8, attr: []const u8) ?[]const u8 {
        const attr_start = std.mem.indexOf(u8, xml, attr) orelse return null;
        const quote_start = attr_start + attr.len;
        if (quote_start >= xml.len) return null;

        const quote_char = xml[quote_start];
        if (quote_char != '"' and quote_char != '\'') return null;

        const value_start = quote_start + 1;
        var value_end = value_start;
        while (value_end < xml.len and xml[value_end] != quote_char) {
            value_end += 1;
        }

        if (value_end > value_start) {
            return xml[value_start..value_end];
        }
        return null;
    }

    /// Get number of rows
    pub fn rowCount(self: *const Self) usize {
        return self.num_rows;
    }

    /// Get number of columns
    pub fn columnCount(self: *const Self) usize {
        return self.num_cols;
    }

    /// Get cell value
    pub fn getCell(self: *const Self, row: usize, col: usize) ?CellValue {
        if (row >= self.cell_values.len) return null;
        if (col >= self.cell_values[row].len) return null;
        return self.cell_values[row][col];
    }

    /// Check if file is valid XLSX (ZIP with correct magic)
    pub fn isValid(data: []const u8) bool {
        if (data.len < 4) return false;
        const magic = std.mem.readInt(u32, data[0..4], .little);
        return magic == ZIP_LOCAL_FILE_MAGIC;
    }
};

// Tests
const testing = std.testing;

test "xlsx: magic validation" {
    const allocator = testing.allocator;

    // Invalid magic
    const bad_data = "NOT A ZIP";
    const result = XlsxReader.init(allocator, bad_data);
    try testing.expectError(error.InvalidZipMagic, result);
}

test "xlsx: isValid check" {
    // Valid ZIP magic
    const valid = [_]u8{ 0x50, 0x4B, 0x03, 0x04 } ++ [_]u8{0} ** 10;
    try testing.expect(XlsxReader.isValid(&valid));

    // Invalid
    try testing.expect(!XlsxReader.isValid("not xlsx"));
}

test "xlsx: read simple fixture" {
    const allocator = testing.allocator;

    // Read uncompressed fixture file (deflate not supported yet)
    const file = std.fs.cwd().openFile("tests/fixtures/simple_uncompressed.xlsx", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 100 * 1024);
    defer allocator.free(data);

    var reader = try XlsxReader.init(allocator, data);
    defer reader.deinit();

    // Should detect rows (6: 1 header + 5 data rows)
    try testing.expectEqual(@as(usize, 6), reader.rowCount());

    // Should detect columns (3: id, name, value)
    try testing.expectEqual(@as(usize, 3), reader.columnCount());

    // Check first row (header)
    const h0 = reader.getCell(0, 0);
    try testing.expect(h0 != null);
    try testing.expectEqual(CellType.inline_string, @as(CellType, h0.?));
    try testing.expectEqualStrings("id", h0.?.inline_string);

    // Check data row
    const d0 = reader.getCell(1, 0); // id = 1
    try testing.expect(d0 != null);
    try testing.expectEqual(CellType.number, @as(CellType, d0.?));
    try testing.expectEqual(@as(f64, 1), d0.?.number);

    const d1 = reader.getCell(1, 1); // name = alice
    try testing.expect(d1 != null);
    try testing.expectEqual(CellType.inline_string, @as(CellType, d1.?));
    try testing.expectEqualStrings("alice", d1.?.inline_string);
}
