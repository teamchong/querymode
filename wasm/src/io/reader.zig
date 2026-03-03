//! Reader interface (VFS abstraction).
//!
//! The Reader interface allows EdgeQ to read data from various sources
//! (files, memory, HTTP) using the same parsing code. This is essential
//! for supporting both native execution and WASM browser targets.

const std = @import("std");

/// Errors that can occur during read operations
pub const ReadError = error{
    /// Requested offset is beyond end of data
    OffsetOutOfBounds,
    /// I/O operation failed
    IoError,
    /// Connection/network error
    NetworkError,
    /// Operation not supported on this platform
    Unsupported,
    /// Generic read failure
    ReadFailed,
    /// Out of memory
    OutOfMemory,
};

/// A byte range for batch reading
pub const ByteRange = struct {
    /// Start offset in file
    start: u64,
    /// End offset (exclusive) in file
    end: u64,

    /// Get the byte length of this range
    pub fn len(self: ByteRange) u64 {
        return self.end - self.start;
    }
};

/// Result of a batch read operation
pub const BatchReadResult = struct {
    /// Array of data buffers, one per requested range
    /// Each buffer is owned by the caller and must be freed
    buffers: [][]u8,
    /// Allocator used for the buffers
    allocator: std.mem.Allocator,

    /// Free all buffers and the result
    pub fn deinit(self: *BatchReadResult) void {
        for (self.buffers) |buf| {
            self.allocator.free(buf);
        }
        self.allocator.free(self.buffers);
    }
};

/// Virtual file system reader interface.
///
/// This type-erased interface allows the same code to read from different
/// sources: local files, memory buffers, or HTTP endpoints.
pub const Reader = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Read data at the given offset into the buffer.
        /// Returns the number of bytes actually read.
        read: *const fn (ptr: *anyopaque, offset: u64, buffer: []u8) ReadError!usize,

        /// Get the total size of the data source.
        size: *const fn (ptr: *anyopaque) ReadError!u64,

        /// Release resources associated with this reader.
        deinit: *const fn (ptr: *anyopaque) void,

        /// Optional batch read for multiple ranges in a single request.
        /// If null, the Reader will fall back to sequential reads.
        /// This enables HTTP multi-range requests for cloud-hosted files.
        batch_read: ?*const fn (ptr: *anyopaque, allocator: std.mem.Allocator, ranges: []const ByteRange) ReadError!BatchReadResult = null,
    };

    /// Read data at the given offset into the buffer.
    ///
    /// Returns the number of bytes actually read, which may be less than
    /// buffer.len if the end of data is reached.
    pub fn read(self: Reader, offset: u64, buffer: []u8) ReadError!usize {
        return self.vtable.read(self.ptr, offset, buffer);
    }

    /// Read exactly buffer.len bytes at the given offset.
    ///
    /// Returns an error if fewer bytes are available.
    pub fn readExact(self: Reader, offset: u64, buffer: []u8) ReadError!void {
        var total_read: usize = 0;
        while (total_read < buffer.len) {
            const bytes_read = try self.read(offset + total_read, buffer[total_read..]);
            if (bytes_read == 0) {
                return ReadError.OffsetOutOfBounds;
            }
            total_read += bytes_read;
        }
    }

    /// Get the total size of the data source in bytes.
    pub fn size(self: Reader) ReadError!u64 {
        return self.vtable.size(self.ptr);
    }

    /// Release resources associated with this reader.
    pub fn deinit(self: Reader) void {
        self.vtable.deinit(self.ptr);
    }

    /// Read and return allocated bytes from the given offset.
    ///
    /// Caller owns the returned memory and must free it.
    pub fn readAlloc(self: Reader, allocator: std.mem.Allocator, offset: u64, len: usize) ![]u8 {
        const buffer = try allocator.alloc(u8, len);
        errdefer allocator.free(buffer);

        try self.readExact(offset, buffer);
        return buffer;
    }

    /// Check if this reader supports batch reads.
    /// If true, batchRead may be more efficient than multiple sequential reads.
    pub fn supportsBatchRead(self: Reader) bool {
        return self.vtable.batch_read != null;
    }

    /// Read multiple byte ranges in a single operation.
    ///
    /// If the underlying reader supports batch reads (e.g., HTTP multi-range),
    /// this will be more efficient. Otherwise, falls back to sequential reads.
    ///
    /// Returns a BatchReadResult with one buffer per range.
    /// Caller must call result.deinit() to free memory.
    pub fn batchRead(self: Reader, allocator: std.mem.Allocator, ranges: []const ByteRange) ReadError!BatchReadResult {
        // Use native batch read if available
        if (self.vtable.batch_read) |batch_fn| {
            return batch_fn(self.ptr, allocator, ranges);
        }

        // Fallback: sequential reads
        return self.batchReadSequential(allocator, ranges);
    }

    /// Fallback implementation using sequential reads
    fn batchReadSequential(self: Reader, allocator: std.mem.Allocator, ranges: []const ByteRange) ReadError!BatchReadResult {
        const buffers = allocator.alloc([]u8, ranges.len) catch return ReadError.OutOfMemory;
        errdefer allocator.free(buffers);

        var completed: usize = 0;
        errdefer {
            for (buffers[0..completed]) |buf| {
                allocator.free(buf);
            }
        }

        for (ranges, 0..) |range, i| {
            const len: usize = @intCast(range.len());
            const buffer = allocator.alloc(u8, len) catch return ReadError.OutOfMemory;
            errdefer allocator.free(buffer);

            try self.readExact(range.start, buffer);
            buffers[i] = buffer;
            completed += 1;
        }

        return BatchReadResult{
            .buffers = buffers,
            .allocator = allocator,
        };
    }
};

test "reader interface" {
    // This test verifies the interface compiles correctly.
    // Actual implementation tests are in file_reader.zig and memory_reader.zig.
    const vtable = Reader.VTable{
        .read = struct {
            fn f(_: *anyopaque, _: u64, _: []u8) ReadError!usize {
                return 0;
            }
        }.f,
        .size = struct {
            fn f(_: *anyopaque) ReadError!u64 {
                return 0;
            }
        }.f,
        .deinit = struct {
            fn f(_: *anyopaque) void {}
        }.f,
    };

    var dummy: u8 = 0;
    const reader_instance = Reader{
        .ptr = @ptrCast(&dummy),
        .vtable = &vtable,
    };

    const sz = try reader_instance.size();
    try std.testing.expectEqual(@as(u64, 0), sz);
}
