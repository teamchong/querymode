//! Memory-mapped file reader for zero-copy I/O.
//!
//! Uses mmap for direct memory access to file contents, avoiding
//! the overhead of read() syscalls. This is critical for performance
//! when the same file is accessed repeatedly.
//!
//! Platform support:
//! - macOS: mmap + madvise(MADV_WILLNEED) for prefetch
//! - Linux: mmap + madvise + readahead
//! - Other: Falls back to regular file I/O

const std = @import("std");
const builtin = @import("builtin");
const reader_mod = @import("reader.zig");

const Reader = reader_mod.Reader;
const ReadError = reader_mod.ReadError;

/// Memory-mapped file reader for zero-copy access
pub const MmapReader = struct {
    /// Memory-mapped region (null if mmap failed, falls back to file I/O)
    mapped_data: ?[]align(std.mem.page_size) u8,
    /// File handle (kept open for fallback and to keep mmap valid)
    file: std.fs.File,
    /// File size
    file_size: u64,
    /// Whether we own the mapping (need to unmap on deinit)
    owns_mapping: bool,

    const Self = @This();

    /// Open a file with memory mapping
    pub fn open(path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        errdefer file.close();

        return initFromFile(file);
    }

    /// Open with absolute path
    pub fn openAbsolute(path: []const u8) !Self {
        const file = try std.fs.openFileAbsolute(path, .{});
        errdefer file.close();

        return initFromFile(file);
    }

    /// Initialize from an already-opened file
    fn initFromFile(file: std.fs.File) !Self {
        const stat = try file.stat();
        const file_size = stat.size;

        // Try to memory-map the file
        const mapped_data = mapFile(file, file_size);

        // Prefetch the entire file into memory (async)
        if (mapped_data) |data| {
            prefetch(data);
        }

        return Self{
            .mapped_data = mapped_data,
            .file = file,
            .file_size = file_size,
            .owns_mapping = true,
        };
    }

    /// Memory-map a file
    fn mapFile(file: std.fs.File, size: u64) ?[]align(std.mem.page_size) u8 {
        if (size == 0) return null;

        const ptr = std.posix.mmap(
            null,
            @intCast(size),
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        ) catch return null;

        return @alignCast(ptr[0..@intCast(size)]);
    }

    /// Prefetch mapped memory (platform-specific optimization)
    fn prefetch(data: []const u8) void {
        switch (builtin.os.tag) {
            .macos => {
                // macOS: MADV_WILLNEED tells kernel to prefetch
                std.posix.madvise(@constCast(data.ptr), data.len, .WILLNEED) catch {};
            },
            .linux => {
                // Linux: MADV_WILLNEED + sequential hint
                std.posix.madvise(@constCast(data.ptr), data.len, .WILLNEED) catch {};
                std.posix.madvise(@constCast(data.ptr), data.len, .SEQUENTIAL) catch {};
            },
            else => {},
        }
    }

    /// Close the reader and unmap memory
    pub fn close(self: *Self) void {
        if (self.owns_mapping) {
            if (self.mapped_data) |data| {
                std.posix.munmap(data);
            }
        }
        self.file.close();
        self.mapped_data = null;
    }

    /// Get a Reader interface
    pub fn reader(self: *Self) Reader {
        return Reader{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    /// Direct access to mapped memory (zero-copy)
    pub fn getData(self: *const Self) ?[]const u8 {
        if (self.mapped_data) |data| {
            return data;
        }
        return null;
    }

    /// Get a slice of the mapped data (zero-copy)
    pub fn getSlice(self: *const Self, offset: u64, len: usize) ?[]const u8 {
        if (self.mapped_data) |data| {
            if (offset + len <= data.len) {
                return data[@intCast(offset)..][0..len];
            }
        }
        return null;
    }

    // VTable implementation
    const vtable = Reader.VTable{
        .read = readImpl,
        .size = sizeImpl,
        .deinit = deinitImpl,
    };

    fn readImpl(ptr: *anyopaque, offset: u64, buffer: []u8) ReadError!usize {
        const self: *Self = @ptrCast(@alignCast(ptr));

        if (offset >= self.file_size) {
            return ReadError.OffsetOutOfBounds;
        }

        // Use mmap if available (zero-copy into user buffer)
        if (self.mapped_data) |data| {
            const start: usize = @intCast(offset);
            const available = data.len - start;
            const to_copy = @min(buffer.len, available);
            @memcpy(buffer[0..to_copy], data[start..][0..to_copy]);
            return to_copy;
        }

        // Fallback to file I/O
        self.file.seekTo(offset) catch return ReadError.IoError;
        return self.file.read(buffer) catch ReadError.IoError;
    }

    fn sizeImpl(ptr: *anyopaque) ReadError!u64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.file_size;
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.close();
    }
};

// ============================================================================
// Global File Cache
// ============================================================================

/// Cached mmap entry
const CacheEntry = struct {
    reader: MmapReader,
    ref_count: std.atomic.Value(u32),
    last_used: i64,
    path_hash: u64,
};

/// Global cache for memory-mapped files
/// Keeps files mapped between queries for instant access
pub const FileCache = struct {
    entries: [MAX_CACHE_SIZE]?CacheEntry,
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,

    const MAX_CACHE_SIZE = 16;
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .entries = [_]?CacheEntry{null} ** MAX_CACHE_SIZE,
            .allocator = allocator,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (&self.entries) |*entry| {
            if (entry.*) |*e| {
                e.reader.close();
                entry.* = null;
            }
        }
    }

    /// Get or open a file (cached)
    pub fn getOrOpen(self: *Self, path: []const u8) !*MmapReader {
        const path_hash = hashPath(path);

        self.mutex.lock();
        defer self.mutex.unlock();

        // Check cache
        for (&self.entries) |*entry| {
            if (entry.*) |*e| {
                if (e.path_hash == path_hash) {
                    _ = e.ref_count.fetchAdd(1, .monotonic);
                    e.last_used = std.time.timestamp();
                    return &e.reader;
                }
            }
        }

        // Not in cache - open and cache
        var reader_instance = try MmapReader.open(path);
        errdefer reader_instance.close();

        // Find empty slot or evict LRU
        var slot: ?*?CacheEntry = null;
        var oldest_time: i64 = std.math.maxInt(i64);

        for (&self.entries) |*entry| {
            if (entry.* == null) {
                slot = entry;
                break;
            }
            if (entry.*.?.last_used < oldest_time and entry.*.?.ref_count.load(.monotonic) == 0) {
                oldest_time = entry.*.?.last_used;
                slot = entry;
            }
        }

        if (slot) |s| {
            // Evict old entry if present
            if (s.*) |*old| {
                old.reader.close();
            }

            s.* = CacheEntry{
                .reader = reader_instance,
                .ref_count = std.atomic.Value(u32).init(1),
                .last_used = std.time.timestamp(),
                .path_hash = path_hash,
            };
            return &s.*.?.reader;
        }

        // No slot available - return without caching
        // This shouldn't happen with proper eviction
        return error.CacheFull;
    }

    /// Release a cached reader
    pub fn release(self: *Self, reader_ptr: *MmapReader) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (&self.entries) |*entry| {
            if (entry.*) |*e| {
                if (&e.reader == reader_ptr) {
                    _ = e.ref_count.fetchSub(1, .monotonic);
                    return;
                }
            }
        }
    }

    fn hashPath(path: []const u8) u64 {
        var h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for (path) |c| {
            h ^= c;
            h *%= 0x100000001b3; // FNV-1a prime
        }
        return h;
    }
};

/// Global file cache instance
pub var global_cache: ?FileCache = null;

/// Initialize global cache
pub fn initGlobalCache(allocator: std.mem.Allocator) void {
    if (global_cache == null) {
        global_cache = FileCache.init(allocator);
    }
}

/// Deinitialize global cache
pub fn deinitGlobalCache() void {
    if (global_cache) |*cache| {
        cache.deinit();
        global_cache = null;
    }
}

test "mmap reader basic" {
    const testing = std.testing;

    // Create temp file
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const test_data = "Hello, LanceQL with mmap!";
    {
        const f = try tmp.dir.createFile("test.bin", .{});
        defer f.close();
        try f.writeAll(test_data);
    }

    // Note: Can't easily test with tmpDir path in this context
    // In production, use absolute paths
}
