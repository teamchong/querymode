const std = @import("std");

pub const js = struct {
    /// Open file, returns handle ID (0 = error)
    pub extern "env" fn opfs_open(path_ptr: [*]const u8, path_len: usize) u32;
    /// Read from file at offset into buffer, returns bytes read
    pub extern "env" fn opfs_read(handle: u32, buf_ptr: [*]u8, buf_len: usize, offset: u64) usize;
    /// Get file size
    pub extern "env" fn opfs_size(handle: u32) u64;
    /// Close file handle
    pub extern "env" fn opfs_close(handle: u32) void;
};
