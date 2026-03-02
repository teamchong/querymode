//! CLI File Utilities
//!
//! File path extraction and Lance dataset reading utilities.

const std = @import("std");

/// Extract table path from SQL query (finds 'path' in FROM clause)
pub fn extractTablePath(query: []const u8) ?[]const u8 {
    // Simple extraction: find FROM 'path' or FROM "path"
    const from_pos = std.mem.indexOf(u8, query, "FROM ") orelse
        std.mem.indexOf(u8, query, "from ") orelse return null;

    const after_from = query[from_pos + 5 ..];

    // Skip whitespace
    var start: usize = 0;
    while (start < after_from.len and (after_from[start] == ' ' or after_from[start] == '\t')) {
        start += 1;
    }

    if (start >= after_from.len) return null;

    // Check for quoted path
    const quote_char = after_from[start];
    if (quote_char == '\'' or quote_char == '"') {
        const path_start = start + 1;
        const path_end = std.mem.indexOfScalarPos(u8, after_from, path_start, quote_char) orelse return null;
        return after_from[path_start..path_end];
    }

    // Unquoted identifier
    var end = start;
    while (end < after_from.len and after_from[end] != ' ' and after_from[end] != '\t' and
        after_from[end] != '\n' and after_from[end] != ';' and after_from[end] != ')')
    {
        end += 1;
    }

    return after_from[start..end];
}

/// Open a file or Lance dataset directory and return its contents
pub fn openFileOrDataset(allocator: std.mem.Allocator, path: []const u8) ?[]const u8 {
    // Check if path is a file or directory
    const stat = std.fs.cwd().statFile(path) catch {
        // Try as directory - read all .lance fragments
        return readLanceDataset(allocator, path);
    };

    if (stat.kind == .directory) {
        // It's a directory, try to open as Lance dataset
        return readLanceDataset(allocator, path);
    }

    // It's a file, open it directly
    var file = std.fs.cwd().openFile(path, .{}) catch return null;
    defer file.close();

    return file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch null;
}

/// Read all .lance fragments from a Lance dataset directory
pub fn readLanceDataset(allocator: std.mem.Allocator, path: []const u8) ?[]const u8 {
    var data_path_buf: [4096]u8 = undefined;
    const data_path = std.fmt.bufPrint(&data_path_buf, "{s}/data", .{path}) catch return null;

    var data_dir = std.fs.cwd().openDir(data_path, .{ .iterate = true }) catch return null;
    defer data_dir.close();

    // Collect all .lance files
    var lance_files = std.ArrayList([]const u8){};
    defer {
        for (lance_files.items) |name| allocator.free(name);
        lance_files.deinit(allocator);
    }

    var iter = data_dir.iterate();
    while (iter.next() catch null) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.name, ".lance")) {
            lance_files.append(allocator, allocator.dupe(u8, entry.name) catch continue) catch continue;
        }
    }

    if (lance_files.items.len == 0) return null;

    // Sort by filename (0.lance, 1.lance, 2.lance, etc.)
    std.mem.sort([]const u8, lance_files.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            // Extract numeric prefix for proper sorting
            const a_num = extractFragmentNumber(a);
            const b_num = extractFragmentNumber(b);
            if (a_num != null and b_num != null) {
                return a_num.? < b_num.?;
            }
            return std.mem.lessThan(u8, a, b);
        }
        fn extractFragmentNumber(name: []const u8) ?u64 {
            // Parse "0.lance", "1.lance", etc.
            const dot_pos = std.mem.indexOf(u8, name, ".") orelse return null;
            return std.fmt.parseInt(u64, name[0..dot_pos], 10) catch null;
        }
    }.lessThan);

    // Read and concatenate all fragments
    var combined = std.ArrayList(u8){};
    for (lance_files.items) |name| {
        var file = data_dir.openFile(name, .{}) catch continue;
        defer file.close();
        const content = file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch continue;
        defer allocator.free(content);
        combined.appendSlice(allocator, content) catch continue;
    }

    if (combined.items.len == 0) return null;
    return combined.toOwnedSlice(allocator) catch null;
}
